pythonversion := "3.12.4"

init:
    echo 'Installing Python 3.12.4.'
    # pyenv install 3.12.4
    pyenv local 3.12.4

    echo 'Creating Python environment.'
    python -m venv venv
    venv/bin/python pip install --upgrade pip
    venv/bin/python pip install -r requirements.txt

    echo "Now run $source ./venv/bin/activate to start venv"

setup-pyenv:
    pyenv local {{pythonversion}}
    python -m venv venv
    venv/bin/pip install --upgrade pip
    venv/bin/pip install -r requirements.txt
    source venv/bin/activate


