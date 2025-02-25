import fs from 'fs';
import path from 'path';
import ffmpeg from 'fluent-ffmpeg';
import { pipeline } from '@xenova/transformers';
import { decode } from 'wav-decoder';

// Configurações
const AUDIO_FILE = './audio.m4a'; // Nome do arquivo de áudio original
const CHUNK_DURATION = 30; // Duração de cada segmento (segundos)
const SPLIT_AUDIO = true; // Define se o áudio deve ser dividido ou não

// Configurações de teste
const TEST_MODE = true;
const MAX_TEST_CHUNKS = 3;

// Função para dividir o áudio em partes menores (ou retornar o arquivo original se SPLIT_AUDIO for falso)
const splitAudio = async (inputFile) => {
    if (!SPLIT_AUDIO) {
        console.log("Processando o áudio inteiro sem dividir...");
        return [inputFile]; // Retorna o áudio inteiro como um único "chunk"
    }

    console.log("Dividindo áudio em partes menores...");
    
    const outputDir = path.join(path.dirname(inputFile), 'chunks');
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir);

    return new Promise((resolve, reject) => {
        ffmpeg(inputFile)
            .output(`${outputDir}/chunk_%03d.wav`)
            .audioCodec('pcm_s16le')
            .audioFrequency(16000)
            .audioChannels(1)
            .format('wav')
            .outputOptions([
                `-f segment`,
                `-segment_time ${CHUNK_DURATION}`,
                `-c:a pcm_s16le`
            ])
            .on('end', () => {
                console.log("Áudio dividido com sucesso.");
                let chunks = fs.readdirSync(outputDir)
                              .map(f => path.join(outputDir, f))
                              .sort();

                if (TEST_MODE) {
                    chunks = chunks.slice(0, MAX_TEST_CHUNKS);
                    console.log(`Modo de teste ativado: processando apenas as ${MAX_TEST_CHUNKS} primeiras partes.`);
                }

                resolve(chunks);
            })
            .on('error', (err) => {
                console.error("Erro ao dividir áudio:", err);
                reject(err);
            })
            .run();
    });
};

// Função para verificar se um segmento contém áudio válido
const isValidAudio = async (filePath) => {
    try {
        const buffer = fs.readFileSync(filePath);
        const audioData = await decode(buffer);
        return audioData.channelData[0].some(sample => Math.abs(sample) > 0.01); // Verifica se há som
    } catch (error) {
        console.error("Erro ao verificar áudio:", error);
        return false;
    }
};

// Função para transcrever um segmento de áudio
const transcribeSegment = async (filePath, model) => {
    if (!(await isValidAudio(filePath))) {
        console.log(`Ignorando ${filePath}, pois contém apenas silêncio.`);
        return "";
    }

    console.log("Transcrevendo segmento:", filePath);
    const buffer = fs.readFileSync(filePath);
    const audioData = await decode(buffer);
    const result = await model(audioData.channelData[0]);

    return result.text;
};

// Função para formatar o texto, inserindo quebras de linha a cada 100 caracteres
const formatTextWithLineBreaks = (text, lineLength = 100) => {
    let formattedText = "";
    let start = 0;

    while (start < text.length) {
        let end = start + lineLength;

        // Se o final ultrapassa o tamanho do texto, ajustamos para o final do texto
        if (end >= text.length) {
            formattedText += text.substring(start);
            break;
        }

        // Se o caractere no limite for um espaço, podemos quebrar ali
        if (text[end] === " ") {
            formattedText += text.substring(start, end) + "\n";
            start = end + 1; // Pula o espaço
        } else {
            // Encontrar o último espaço antes do limite
            let lastSpace = text.lastIndexOf(" ", end);
            if (lastSpace > start) {
                formattedText += text.substring(start, lastSpace) + "\n";
                start = lastSpace + 1;
            } else {
                // Se não houver espaço antes do limite, quebra no limite mesmo
                formattedText += text.substring(start, end) + "\n";
                start = end;
            }
        }
    }

    return formattedText;
};

// Função principal
const main = async () => {
    try {
        console.log("Carregando modelo...");
        const model = await pipeline('automatic-speech-recognition', 'Xenova/whisper-large-v3');

        console.log("Dividindo áudio...");
        const chunks = await splitAudio(AUDIO_FILE);

        console.log("Iniciando transcrição...");
        let fullText = "";

        for (let chunk of chunks) {
            const text = await transcribeSegment(chunk, model);
            if (text.trim()) {
                fullText += formatTextWithLineBreaks(text) + "\n"; // Aplica a formatação e adiciona espaçamento entre segmentos
            }
        }

        // Salvar transcrição final com formatação correta
        const outputTextFile = AUDIO_FILE.replace(path.extname(AUDIO_FILE), '.txt');
        fs.writeFileSync(outputTextFile, fullText.trim(), 'utf8');

        console.log("Transcrição concluída! Salva em:", outputTextFile);
    } catch (error) {
        console.error("Erro:", error);
    }
};

main();
