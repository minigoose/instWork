import json
import pymysql
import pprint
from flatten_json import flatten

pp = pprint.PrettyPrinter(indent=4)

def list_append(row, ssp, device, value, tier):
	# append SSP
	row.append(ssp)
	# # append Device
	row.append(device)
	# # after floor
	row.append(value)
	# # tier
	row.append(tier)

def ssp_device_tier(count, key):
	if count == 0:
		if 'webfloor_after' in key:
			ssp = key.split('_')[0][:-8].lower()
			device = 'web'
			tier = 'all_tier'
		elif 'mobilefloor_after' in key:
			ssp = key.split('_')[0][:-11].lower()
			device = 'mobile'
			tier = 'all_tier'
		elif ('floorsjson' in key and 'after' in key):
			ssp = key.split('_')[3]
			device = key.split('_')[4]
			tier = 'tier_' + key.split('_')[2]
	else:
		ssp = key.split('_')[1]
		device = key.split('_')[2]
		tier = 'all_tier'
	return ssp, device, tier


def SQL():

	SQL_list = []

	count = 0
	for database in ["EMBED", "EMBED"]:
		count = 0
		# "EMBED", "EVERTEST", 
		if count == 0:
			SQL = "SELECT SUBSTR(l.createdAt, 1, 10) as Date, substr(l.createdAt, 12, 8) as Time, l.fieldName as Site, s.siteUUID, l.changes   \
					from LOG as l join SITE as s  \
					where \
   					(l.referenceUUID = s.adUUID) and \
   					(l.changes like '%FloorsJSON%' or l.changes like '%WebFloor%' or l.changes like '%MobileFloor%');;"
		elif count == 1:
			SQL = "SELECT pf.date, s.siteURL, s.siteUUID, pf.postBidFloors    \
					from POSTBIDFLOORS as pf   \
					JOIN SITE as s using (siteUUID);"

		db = pymysql.connect(host="prod-read-replica.cpbybmeoadzj.us-east-1.rds.amazonaws.com",    # your host, usually localhost
								user="readonly",         # your username
								passwd="rdslionking12",  # your password
								db=database)        # name of the data base

		cur = db.cursor()

		cur.execute(SQL)

		for row in cur.fetchall():
			row = list(row)
			if count == 0:
				parsed_json = json.loads(row[4])
				del row[4]
			else:
				parsed_json = json.loads(row[3])
				del row[3]

			flatten_json = flatten(parsed_json)
			row_copy = row[:]

			for key, value in flatten_json.items():
				key = key.lower()
				if count == 0:
					if ('webfloor_after' in key) or ('mobilefloor_after' in key) or ('floorsjson' in key and 'after' in key):
						
						list_append(row_copy, ssp_device_tier(count, key)[0], ssp_device_tier(count, key)[1], value, ssp_device_tier(count, key)[2])

						pp.pprint(row_copy)

						row_copy = row[:]

				else:
					time = key.split('_')[0]
					if len(time) == 1:
						time = '0' + time 
					# insert time
					row_copy.insert(1, time + ':00:00')

					list_append(row_copy, ssp_device_tier(count, key)[0], ssp_device_tier(count, key)[1], value, ssp_device_tier(count, key)[2])

					row_copy[0] = row_copy[0].strftime('%Y-%m-%d')

					pp.pprint (row_copy)

					row_copy = row[:]

		count += 1

	return

test = SQL()
