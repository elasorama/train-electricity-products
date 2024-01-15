import urllib.request
import json
import os
import ssl
import dotenv
import datetime
import math
import time

SECONDS_IN_DAY = 24 * 60 * 60

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Loading the .env 
dotenv.load_dotenv()

# Getting the current timestamp
now = datetime.datetime.now()

# calculating the seconds from midnight 
seconds_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

# Calculating the sin and cos features
seconds_since_midnight_sin = math.sin(2 * math.pi * seconds_since_midnight / SECONDS_IN_DAY)
seconds_since_midnight_cos = math.cos(2 * math.pi * seconds_since_midnight / SECONDS_IN_DAY)

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data =  {
  "input_data": {
    "columns": [
      "seconds_since_midnight_sin",
      "seconds_since_midnight_cos"
    ],
    "data": [
      [
        seconds_since_midnight_sin,
        seconds_since_midnight_cos
      ]
    ]
  },
  "params": {}
}

body = str.encode(json.dumps(data))

url = 'https://power-usage.swedencentral.inference.ml.azure.com/score'
# Replace this with the primary/secondary key or AMLToken for the endpoint
api_key = os.getenv("API_KEY")
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'power-usage-prediction-1' }

req = urllib.request.Request(url, body, headers)

try:
    start = time.time()
    response = urllib.request.urlopen(req)
    response_time = time.time() - start

    result = response.read()
    print(result)

    # Printing the time took in miliseconds
    print(f"Response time: {response_time * 1000} ms")

except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
