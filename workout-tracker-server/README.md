# workout-tracker-server
A simple flask server to handle workout tracking requests

usage:
 - install requirements.txt
 - set the following environment variables:
    - FLASK_APP = workout-tracker-server
    - FLASK_ENV = development
    - for mac/linux:
      - `export <var name>=<var value>`
    - for windows cmd:
      - `set <var name>=<var value>`
    - for windows powershell:
      - `$env:<var name> = "<var value>"`
 - `flask run`
 - go to http://localhost:5000/hello
