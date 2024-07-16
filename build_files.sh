python3.9 -m venv venv

# activate the virtual environment
source venv/bin/activate
pip install -r requirements.txt 
python3.8 manage.py collectstatic
