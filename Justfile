
init:
    echo 'Installing Python 3.12.4.'
    # pyenv install 3.12.4
    pyenv global 3.12.4

    echo 'Creating Python environment.'
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
