# Future_Bag

## Set the environment
```
pip install -U openai-whisper

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirement.txt
```

## Set the google cloud service
In windows run code in the terminals:
```
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
$env:Temp\GoogleCloudSDKInstaller.exe
set GOOGLE_APPLICATION_CREDENTIALS=future_bag.json
```

## Workflow
- The **demo.py** file shows three constant images in the folder **demo_images/** in VR.
- The **audio2image.py** generates images according to the users saying and shows in VR.

1. Set the constants for the file

2. Set up the openAI API and Google Cloud credentials:
```
- Go to Google Cloud Console
- IAM & Admin Setting
- Service Accounts
- There is a service that you have create before for future bag
- Klick Actions -> Manage Keys
- Klick ADD KEY -> Json file
- Rename it as credentials.json as stored in the root directory
```
3. Trigger words settings are in lines 211-217. If you want to use trigger words uncomment the part you want. Edit or add more trigger words in the word list in line 213.
4. GPT enhanced prompts settings are in line 219. If you do not want to use gpt to change the prompt.
5. How GPT changes the prompts are in function are written in function **enhance_description()**. If you want to change the GPT's behavior, change the content from line 265 to 268.