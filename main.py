import os
import torch
import ffmpeg
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

# Configurações
AUDIO_FILE = "audio.m4a"  # Nome do arquivo de áudio original
CHUNK_DURATION = 15  # Duração de cada segmento (segundos)
SPLIT_AUDIO = True  # Define se o áudio deve ser dividido
TEST_MODE = False  # Modo de teste
MAX_TEST_CHUNKS = 3  # Máximo de chunks para teste

# Verifica se a GPU está disponível
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# Criar modelo Whisper acelerado por GPU
model = WhisperModel("large-v3", device=DEVICE, compute_type="float16")


def split_audio(input_file):
    """ Divide o áudio em segmentos menores usando FFmpeg com aceleração CUDA """
    if not SPLIT_AUDIO:
        print("Processando áudio inteiro sem dividir...")
        return [input_file]

    print("Dividindo áudio em partes menores...")
    output_dir = os.path.join(os.path.dirname(input_file), "chunks")
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, "chunk_%03d.wav")

    try:
        (
            ffmpeg.input(input_file)
            .output(
                output_pattern,
                acodec="pcm_s16le",
                ar="16000",
                ac=1,
                f="segment",
                segment_time=CHUNK_DURATION
            )
            .run(quiet=True)
        )
        chunks = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".wav")])

        if TEST_MODE:
            chunks = chunks[:MAX_TEST_CHUNKS]
            print(f"Modo de teste ativado: processando apenas {MAX_TEST_CHUNKS} segmentos.")

        return chunks
    except ffmpeg.Error as e:
        print(f"Erro ao dividir áudio: {e.stderr.decode()}")
        return []


def is_valid_audio(file_path):
    """ Verifica se o arquivo de áudio contém som significativo """
    try:
        audio, _ = sf.read(file_path)
        return np.any(np.abs(audio) > 0.01)  # Verifica se há som relevante
    except Exception as e:
        print(f"Erro ao verificar áudio: {e}")
        return False


def transcribe_segment(file_path, model):
    """ Transcreve um segmento de áudio se for válido """
    if not is_valid_audio(file_path):
        print(f"Ignorando {file_path}, pois contém apenas silêncio.")
        return ""

    print(f"Transcrevendo segmento: {file_path}")
    segments, _ = model.transcribe(file_path)
    return " ".join(segment.text for segment in segments)


def format_text_with_line_breaks(text, line_length=100):
    """ Formata o texto inserindo quebras de linha a cada N caracteres """
    formatted_text = ""
    start = 0
    while start < len(text):
        end = min(start + line_length, len(text))
        last_space = text.rfind(" ", start, end)
        if last_space != -1:
            formatted_text += text[start:last_space] + "\n"
            start = last_space + 1
        else:
            formatted_text += text[start:end] + "\n"
            start = end
    return formatted_text


def main():
    try:
        print("Dividindo áudio...")
        chunks = split_audio(AUDIO_FILE)

        print("Iniciando transcrição...")
        full_text = ""

        for chunk in chunks:
            text = transcribe_segment(chunk, model)
            if text.strip():
                full_text += format_text_with_line_breaks(text) + "\n"

        output_text_file = AUDIO_FILE.replace(os.path.splitext(AUDIO_FILE)[1], ".txt")
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(full_text.strip())

        print(f"Transcrição concluída! Salva em: {output_text_file}")

    except Exception as error:
        print(f"Erro: {error}")


if __name__ == "__main__":
    main()
