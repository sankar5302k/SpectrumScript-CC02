# SpectrumScript-CC02

Step-1: Create Virtual Environment (using CMD):
1. Change the Directory where you want to create an Virtual Environment in CMD.
2. IN CMD use this code to create a Virtual Environment  - *py -m venv myworld*.

Step-2: Install packages in Virtual Environment:
1. Change directory into ```myworld/scripts```  in CMD.
2. Paste the requirements.txt file into ```myworld/scripts```.
3. IN CMD use this code to install packages - *pip install -r requirements.txt*.

Step-3: Migrate the folder:
1. Download  SpectrumScript-CC02 Folder and paste in the same folder or near where Virtual Environment is created.
2.  Change Directory to ```SpectrumScript-CC02/```.
3. IN CMD use this code to Migrate - *py manage.py migrate* and *py manage.py collectstatic*.

Step-4: Run the server:
1. Change Directory to ```SpectrumScript-CC02/``` (if needed).
2.  IN CMD use this code to Runserver - *py manage.py runserver*.
3. Paste the localhost into a browser.
