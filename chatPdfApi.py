import requests
import json
import re


class chatPdfApi:


    def upload_pdf(temp_pdf_path):
        api_key = "sec_sS2UCv3IBiTYh8oCrRSo4ITeyKTCm2xd"
        files = [
            ('file', ('file', open(temp_pdf_path, 'rb'), 'application/octet-stream'))
        ]
        headers = {
            'x-api-key': api_key
        }

        response = requests.post(
            'https://api.chatpdf.com/v1/sources/add-file', headers=headers, files=files)

        if response.status_code == 200:
            print('Source ID:', response.json()['sourceId'])
            return response.json()['sourceId']
        else:
            print('Status:', response.status_code)
            print('Error:', response.text)


    def ask_question(source_id, prompt):
        api_key = "sec_sS2UCv3IBiTYh8oCrRSo4ITeyKTCm2xd"
        headers = {
            'x-api-key': api_key,
            "Content-Type": "application/json",
        }

        data = {
            'sourceId': source_id,
            'messages': [
                {
                    'role': "user",
                    'content': prompt,
                }
            ]
        }

        response = requests.post(
            'https://api.chatpdf.com/v1/chats/message', headers=headers, json=data)

        if response.status_code == 200:
            print('Result:', response.json()['content'])
            return response.json()
        else:
            print('Status:', response.status_code)
            print('Error:', response.text)

    def generate_questions(source_list):

        qa_dict = {}
        queries_dict = {}

        with open('promt.txt', 'r') as file:
            # Read the contents of the file
            prompt = file.read()

            for source_id in source_list:
                response_json = chatPdfApi.ask_question(source_id, prompt)
                try:
                    data_list = json.loads(response_json["content"])
                    queries_dict = {item['Question']: item['Answer'] for item in data_list}
                except json.JSONDecodeError as e:
                    print("JSON decoding error:", str(e))

                qa_dict.update(queries_dict)

        # Save qa_dict to txt file
        with open('qa_dict.txt', 'w') as qa_file:
            qa_file.write(json.dumps(qa_dict, indent=4))  # Write qa_dict to a file in JSON format

        return qa_dict