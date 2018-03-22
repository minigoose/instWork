import requests
import pprint

pp = pprint.PrettyPrinter(indent=4)

api_call = "https://api.keen.io/3.0/projects/56ddffe896773d7e98d63393/queries/count?api_key=&event_collection=answerQuestion&group_by=%5B%22questionUUID%22%2C%22request.embedUUID%22%5D&timezone=US%2FEastern&timeframe=this_14_days&filters=%5B%5D"

response = requests.post(api_call)

keen = response.json()


pp.pprint (keen)