# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Pergunta ao Gemini (API do Google) sobre vídeos e imagens e imprime a resposta em texto.

    uv run ferramentas/gemini.py <arquivo> [<arquivo>…] "<pergunta>"

Chaves no .env da raiz: GEMINI_API_KEY e GEMINI_MODEL (ex.: um gemini-*-flash). API nativa (generateContent): o vídeo vai
inline (base64), reduzido a 640 px de altura (o Gemini amostra ~1 quadro/s e ouve o áudio); um vídeo com menos de 1 s vai
como o quadro do meio (o Gemini recusa vídeo curto demais); imagem vai inline. HTTP direto. Também importado pelo
banco.py e pelo calibragem.py (que montam as mensagens com `bloco` e chamam `chamar`).
"""
import base64
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from base import env, ffmpeg, inspecionar  # noqa: E402

URL = 'https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent'
PENSAR = 1024  # orçamento de raciocínio (tokens), somado ao limite de saída pedido: o raciocínio não come a resposta
EXT_VIDEO = ('.mp4', '.mov', '.m4v', '.webm', '.mkv')
EXT_IMAGEM = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp', '.gif': 'image/gif'}
MIN_VIDEO = 1.0  # do 18: direcao.MIN_VIDEO_INSERT (abaixo disso o Gemini responde "Provider returned error")
MAX_IMAGEM = 4 * 1024 * 1024  # imagens maiores vão reduzidas (uma página inteira capturada passa fácil disso)
TIMEOUT = 180


def _inline(arq: Path, mime: str) -> dict:
    return {'inline_data': {'mime_type': mime, 'data': base64.b64encode(arq.read_bytes()).decode()}}


def bloco(arq: Path, tmp: Path, preparado: bool = False) -> dict:
    """Um arquivo como parte da mensagem (inline_data do Gemini). `preparado`: o vídeo já está pequeno (vai como está)."""
    arq = Path(arq)
    if not arq.is_file():
        raise SystemExit(f'Arquivo não encontrado: {arq}')
    ext = arq.suffix.lower()
    if ext in EXT_IMAGEM:
        if arq.stat().st_size <= MAX_IMAGEM:
            return _inline(arq, EXT_IMAGEM[ext])
        menor = tmp / f'{arq.stem}.jpg'
        ffmpeg('-i', str(arq), '-vf', "scale='if(gt(iw,ih),min(1600,iw),-2)':'if(gt(iw,ih),-2,min(1600,ih))'", '-q:v', '4', str(menor))
        return _inline(menor, 'image/jpeg')
    if ext not in EXT_VIDEO:
        raise SystemExit(f'Formato não aceito ({ext or "sem extensão"}): use vídeo (MP4, MOV, WebM) ou imagem (JPG, PNG, WebP).')
    if preparado:
        return _inline(arq, 'video/mp4')
    info = inspecionar(arq)
    if info['duracao'] < MIN_VIDEO:  # praticamente uma imagem parada: vai o quadro do meio
        quadro = tmp / f'{arq.stem}_meio.jpg'
        ffmpeg('-ss', f'{info["duracao"] / 2:.3f}', '-i', str(arq), '-frames:v', '1', '-vf', 'scale=-2:640', '-q:v', '4', str(quadro))
        return _inline(quadro, 'image/jpeg')
    leve = tmp / f'{arq.stem}_640.mp4'  # do 18: direcao._clipe
    audio = ['-c:a', 'aac', '-b:a', '64k'] if info['tem_audio'] else ['-an']
    ffmpeg('-i', str(arq), '-vf', 'scale=-2:640', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', '-pix_fmt', 'yuv420p', *audio, str(leve))
    return _inline(leve, 'video/mp4')


def _partes(conteudo) -> list[dict]:
    """O conteúdo de uma mensagem ({'type': 'text'} ou o que o `bloco` devolve) como partes do Gemini."""
    if isinstance(conteudo, str):
        return [{'text': conteudo}]
    return [{'text': c['text']} if c.get('type') == 'text' else c for c in conteudo]


def chamar(mensagens: list[dict], esquema: dict | None = None, max_tokens: int = 4000, timeout: float = TIMEOUT,
           uso: dict | None = None):
    """Uma chamada ao Gemini (modelo do .env, temperatura 0, raciocínio curto). Com `esquema` (JSON Schema), a resposta
    vem estruturada e é devolvida já lida (dict); senão, texto. Até 3 novas tentativas em falha de rede, 429 ou 5xx (esperando 2, 10 e 30 s).
    `uso` (um dict), se vier, recebe os tokens (`total_tokens`)."""
    sistema = [p for m in mensagens if m['role'] == 'system' for p in _partes(m['content'])]
    config = {'temperature': 0, 'maxOutputTokens': max_tokens + PENSAR, 'thinkingConfig': {'thinkingBudget': PENSAR}}
    if esquema:
        config |= {'responseMimeType': 'application/json', 'responseJsonSchema': esquema}
    corpo = {'contents': [{'role': 'user', 'parts': [p for m in mensagens if m['role'] == 'user' for p in _partes(m['content'])]}],
             'generationConfig': config, **({'systemInstruction': {'parts': sistema}} if sistema else {})}
    pedido = urllib.request.Request(URL.format(modelo=env('GEMINI_MODEL')), data=json.dumps(corpo).encode(), method='POST', headers={
        'x-goog-api-key': env('GEMINI_API_KEY'), 'Content-Type': 'application/json'})
    for tentativa in (1, 2, 3, 4):
        try:
            with urllib.request.urlopen(pedido, timeout=timeout) as r:
                dados = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            texto = e.read().decode(errors='replace')[:400]
            if (e.code == 429 or e.code >= 500) and tentativa < 4:  # limite por minuto e pico de demanda passam sozinhos
                time.sleep((2, 10, 30)[tentativa - 1])
                continue
            if e.code == 429:
                raise SystemExit('Sem crédito (cota) no Gemini. Veja o plano da chave e tente de novo.')
            raise SystemExit(f'O Gemini recusou o pedido (HTTP {e.code}): {texto}')
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if tentativa < 4:
                time.sleep(2)
                continue
            raise SystemExit(f'Sem resposta do Gemini ({getattr(e, "reason", e)}); tente de novo.')
    if bloqueio := (dados.get('promptFeedback') or {}).get('blockReason'):
        raise SystemExit(f'O Gemini bloqueou o pedido: {bloqueio}')
    if uso is not None:
        uso['total_tokens'] = (dados.get('usageMetadata') or {}).get('totalTokenCount')
    partes = ((dados.get('candidates') or [{}])[0].get('content') or {}).get('parts') or []
    conteudo = ''.join(p.get('text', '') for p in partes if not p.get('thought'))
    if esquema:
        try:
            return json.loads(conteudo.strip().removeprefix('```json').removesuffix('```'))
        except ValueError:
            raise SystemExit(f'Resposta fora do formato pedido: {conteudo[:300]}')
    return conteudo.strip()


def perguntar(arquivos: list, pergunta: str, sistema: str | None = None, esquema: dict | None = None,
              preparados: bool = False, max_tokens: int = 4000):
    """Manda os arquivos e a pergunta; devolve o texto (ou o dict, com `esquema`)."""
    with tempfile.TemporaryDirectory() as tmp:
        conteudo = [{'type': 'text', 'text': pergunta}] + [bloco(Path(a), Path(tmp), preparados) for a in arquivos]
        mensagens = ([{'role': 'system', 'content': sistema}] if sistema else []) + [{'role': 'user', 'content': conteudo}]
        return chamar(mensagens, esquema, max_tokens)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    print(perguntar(sys.argv[1:-1], sys.argv[-1]))
