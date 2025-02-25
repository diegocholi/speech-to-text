# ⚠️ Certifique-se de que o FFmpeg está instalado!

- Windows: [Baixe aqui](https://ffmpeg.org/download.html) e adicione ao PATH.
- Linux/macOS: Instale com:

```sh
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

# Configure venv (Use pythyn ou Node. Python é mais performatico com placa de video)

```sh
python3 -m venv venv
```

```sh
source venv/bin/activate # No Windows: venv\Scripts\activate
```

```sh
pip install -r requirements.txt
```

## Automatizar vidia-cudnn-cu12 para toda vez que ativar o venv

- Execute o comando abaixo com o venv ativo para apontar para o nvidia-cudnn-cu12 correto:

```
echo 'export CUDNN_LIBRARY=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib' >> venv/bin/activate
echo 'export CUDNN_INCLUDE_DIR=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/include' >> venv/bin/activate
echo 'export LD_LIBRARY_PATH=$CUDNN_LIBRARY:$LD_LIBRARY_PATH' >> venv/bin/activate
```

# Dependencias python (já estão todas no requirements.txt com faster-whisper)

- Dependencias com faster-whisper

```sh
pip install torch torchaudio soundfile numpy ffmpeg-python faster-whisper
```

- Instale o cuDNN via pip

```sh
pip install nvidia-cudnn-cu12==9.1.0.70
```

- Se quiser rodar o modelo original da OpenAI em vez do faster-whisper:

```sh
pip install torch torchaudio soundfile numpy ffmpeg-python openai-whisper
```
