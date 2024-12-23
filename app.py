from flask import Flask, request, jsonify
from urllib.parse import urlparse

from transformers import pipeline
import requests
from langdetect import detect
import torch

app = Flask(__name__)

# Check if GPU is available and set device accordingly
device = 0 if torch.cuda.is_available() else -1

asr_pipeline = pipeline("automatic-speech-recognition", model="rowjak/whisper-tiny-minds14-en", device=device)
classification_pipeline = pipeline("audio-classification", model="rowjak/wav2vec2-minds14-audio-classification-all", device=device)

def download_audio(url, filename="audio.mp3"):
    response = requests.get(url)
    with open(filename, "wb") as file:
        file.write(response.content)
    return filename


@app.route('/webhook', methods=['POST'])
def webhook():
    # Mengambil data JSON dari request
    data = request.get_json()

    # Memeriksa apakah payload ada dalam data
    if 'payload' not in data:
        return jsonify({'error': 'Payload not found'}), 400


    if data['payload'].get('hasMedia', None) == True:

        media_data = data['payload'].get('media', {})

        url = media_data.get('url', None)


        parsed_url = urlparse(url).path

        filename = parsed_url.split('/')[-1]

        if filename.endswith(('.wav', '.mp3', '.oga')):
            audio_url = "http://xxx/inspira/NLP-C/" + filename
            audio_file = download_audio(audio_url)

            # Transcribe and detect language
            transcription = asr_pipeline(audio_file)
            transcribed_text = transcription['text']
            detected_language = detect(transcribed_text)

            classification_res = classification_pipeline(audio_file)

            hasil = f"Lang : {detected_language} \nText : {transcribed_text}\nClasification : {classification_res[0]['label']} ({classification_res[0]['score']})"

            # print("Original Text:", transcribed_text)
            # print("Detected Language:", detected_language)

            from_number = data['payload'].get('from', None)

            response = requests.post(
                'https://xxx/api/sendText',
                headers={
                    'Content-Type': 'application/json; charset=utf-8',
                    'Accept': 'application/json',
                    'X-Api-Key': 'xxx'
                },
                json={  # Menggunakan parameter `json` untuk mengirim raw JSON
                    'chatId': from_number,
                    'text': hasil,
                    'session': 'NLP-C'
                },
                verify=False  # Ini sesuai dengan withoutVerifying() di PHP
            )
            # Mengembalikan respons dari permintaan HTTP
            return jsonify(response.json()), response.status_code

        else:
            from_number = data['payload'].get('from', None)

            response = requests.post(
                'https://xxx/api/sendText',
                headers={
                    'Content-Type': 'application/json; charset=utf-8',
                    'Accept': 'application/json',
                    'X-Api-Key': 'xxx'
                },
                json={  # Menggunakan parameter `json` untuk mengirim raw JSON
                    'chatId': from_number,
                    'text': "Anda Tidak Mengirim File/ Ekstensinya bukan .wav/.mp3",
                    'session': 'NLP-C'
                },
                verify=False  # Ini sesuai dengan withoutVerifying() di PHP
            )

            return jsonify(response.json()), response.status_code

        # Audio URL


    else:


        from_number = data['payload'].get('from', None)

        response = requests.post(
            'https://xxx/api/sendText',
            headers={
                'Content-Type': 'application/json; charset=utf-8',
                'Accept': 'application/json',
                'X-Api-Key': 'xxx'
            },
            json={  # Menggunakan parameter `json` untuk mengirim raw JSON
                'chatId': from_number,
                'text': "Anda Tidak Mengirim File/ Ekstensinya bukan .wav/.mp3",
                'session': 'NLP-C'
            },
            verify=False  # Ini sesuai dengan withoutVerifying() di PHP
        )

        return jsonify(response.json()), response.status_code


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

