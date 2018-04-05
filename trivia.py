import pymysql
import csv

def question():
# def First_Day_Revenue(date):
    db = pymysql.connect(host="prod-read-replica.cpbybmeoadzj.us-east-1.rds.amazonaws.com",    # your host, usually localhost
                        user="readonly",         # your username
                        passwd="rdslionking12",  # your password
                        db="EVERTEST")        # name of the data base

    # you must create a Cursor object. It will let
    #  you execute all the queries you need
    cur = db.cursor()

    # Use all the SQL you like
    cur.execute("SELECT p.PredictionUUID, p.PredictionTitle from PREDICTIONS as p")

    with open("C:\Users\Nick Liu\Desktop\\trivia111.csv", "wb") as f:
        writer = csv.writer(f)
        for val in cur:
            writer.writerow(val)
    print ("Writing complete")


a = question()