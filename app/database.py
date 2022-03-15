import pymongo

class DB:

    def __init__(self,host,port,username,password):

        #instantiate client
        self.myclient = pymongo.MongoClient("mongodb://{}:{}/".format(host,port),username=username,password=password)
        #create database if its not exist
        self.mydb = self.myclient["UTAC"]
        #create collection if its not exist
        self.Ct3hours_col = self.mydb["chemical tin 3 hours"]
        self.Ct24hours_col = self.mydb["chemical tin 24 hours"]
        self.Ct168hours_col = self.mydb["chemical tin 168 hours"]

        self.St3hours_col = self.mydb["solder thickness 3 hours"]
        self.St24hours_col = self.mydb["solder thickness 24 hours"]
        self.St168hours_col = self.mydb["solder thickness 168 hours"]

        self.ct_gtruth = self.mydb["chemical tin groud truth"]
        self.st_gtruth = self.mydb["solder thickness groud truth"]