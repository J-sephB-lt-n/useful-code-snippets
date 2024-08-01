```
TAGS: TO|DO
DESCRIPTION: TODO
```

```bash
$
.
├── deploy_lambda_func.sh
├── lambda_function.py
└── requirements.txt tree
```

```bash
# deploy_lambda_func.sh
mkdir package
pip install -r requirements.txt --target ./package
cd package || exit
zip -r ../deploy_pkg.zip .
cd ..
zip deploy_pkg.zip lambda_function.py
aws lambda update-function-code --function-name YourLambdaFuncNameOnAws --zip-file fileb://deploy_pkg.zip
rm deploy_pkg.zip
rm -r package
```

```python
# lambda_function.python

# imports go here

def lambda_handler(event, context):
    print(event)  # event contains the request body
    # do some thing here
    return {"statusCode": 200, "body": "OK"}
```
