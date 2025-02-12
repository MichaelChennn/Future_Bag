# Setting Up the google cloud storage

- Log in the google account
- Create a new project called **future_bag**
- Get in the **Cloud Storage**
- Select **Buckets**
- Create a **Buckets**
```
name: future_bag
region_type: multi-region eu
storage class: default class, Standard
control access: do not prevent public access!!!
control access: uniform
protect object data: soft delete policy -> set custom      retention duration 0 days
```
- Click **PREMISSIONS** 
- Click **GRANT ACCESS**
```
- New principals: allUsers
- Assign roles: Storage Legacy Bucket Owner
```
- Click **SAVE**
- Click **ALLOW PUBLIC ACCESS**

- Get in **IAM & Admin**
- Click **Service Accounts**
- Click **Create service account**
```
Service account name: future_bag_bucket
Service account ID: future-bag-bucket
Service account description: upload the generated images
Select a role: Owner
Service account users role: ipd@ed.tum.de
Service account admins role: ipd@ed.tum.de
```
- Click the service accounts you just created
- Click **KEYS**
- Click **ADD KEY**
- Click **CREATE NEW KEY**
- Click **JSON**
- Click **CREATE**
- Move the downloaded json file to the root directory
- Rename it as '**credentials.json**'
