import requests
import json
import csv
from datetime import datetime
import getpass
import sys
from pymongo import MongoClient

startDate = '2018-02-15'

endDate = '2018-02-28'

###################################################### Helper Functions ###########################################################

def formatPageViewsData(queryData):
	formatted_pageViewsData = {}
	for entry in queryData:
		siteUUID = entry['_id']['siteUUID']
		value = entry['embedPageViews']
		formatted_pageViewsData.update({siteUUID : value})
	return formatted_pageViewsData

def convertPageViewsDataToCsv(formatted_pageViewsData):
	with open('pageViews' + startDate +'_to_'+ endDate + '.csv', 'wb') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		headers = ['siteUUID', 'pageViews']
		filewriter.writerow(headers)
		for siteUUID in formatted_pageViewsData:
			pageViewsRow = []
			pageViewsRow.append(siteUUID)	
			pageViewsRow.append(formatted_pageViewsData[siteUUID])
			filewriter.writerow(pageViewsRow)
	return

###################################################### Options ###########################################################

def getPageViewsData():
	# global startDate
	# global endDate
	# print "Enter start date:"
	# startDate = raw_input()
	# print "Enter end date:"
	# endDate = raw_input()

	# Connect to production database
	client = MongoClient('ds015876-a0.mlab.com', 15876)
	db = client.analysis
	db.authenticate('analysist001', 'analysistrocks321', source='analysis')

	collections = client.analysis.revenuedata
	
	pipeline = [{'$match':{'date':{'$gte':startDate,'$lte': endDate}}},{'$group':{'_id':{'siteUUID':'$siteUUID'}, 'embedPageViews':{'$sum':'$embedPageViews'}}}]

	# Fire query to get data
	pageViewsData = collections.aggregate(pipeline)

	# Format query results
	formatted_pageViewsData = formatPageViewsData(pageViewsData)


	# Generate CSV for data
	convertPageViewsDataToCsv(formatted_pageViewsData)

###################################################### Main call ###########################################################

getPageViewsData()