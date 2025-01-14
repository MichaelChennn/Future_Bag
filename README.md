# Future_Bag

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
