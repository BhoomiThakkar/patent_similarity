import re

def cleanpatent(id):
    id = str(id)
    id = re.sub(r'[^0-9]', '', id)
    return id

def clean_name(x):
    x=re.sub(r'\s+',' ',x)
    x=re.sub(r'[^[a-zA-Z ]','',x)
    return x