import sqlite3 as sql
import random
import datetime
connection = sql.connect('MOA.db')



class Job:
    def __init__(self,job_id,mode,status):
        self.job_id = job_id
        self.mode = mode
        self.status = status

def insertJobs(Jobid, n_tasks, mode, n_classes, graph_conv_layers, activation, number_atom_features, residual, batchnorm, dropout, predictor_hidden_feats, predictor_dropout, self_loop, currentstatus, datasetFilename, resultfilename):
    print(str(Jobid)+str(n_tasks)+str(mode)+str(n_classes)+str(graph_conv_layers)+str(activation)+str(number_atom_features)+str(residual)+str(batchnorm)+str(dropout)+str(predictor_hidden_feats)+str(predictor_dropout)+str(self_loop)+str(currentstatus)+str(datasetFilename)+str(resultfilename))
    con = sql.connect("MOA.db")
    cur = con.cursor()
    cur.execute("Insert INTO GCNModel (Jobid, n_tasks, mode, n_classes, graph_conv_layers, activation, number_atom_features, residual, batchnorm, dropout, predictor_hidden_feats, predictor_dropout, self_loop, currentstatus, datasetFilename, resultfilename) Values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)" , (Jobid, n_tasks, mode, n_classes, graph_conv_layers, activation, number_atom_features, residual, batchnorm, dropout, predictor_hidden_feats, predictor_dropout, self_loop, currentstatus, datasetFilename, resultfilename))
    con.commit()
    con.close()
    return Jobid

def insertJobsGAT(Jobid, n_tasks, mode, n_classes, att_hd,dropo,pred_hd_feat,pred_drp,alpha,self_loop,graphcn,graphat_aggmode,currentstatus, datasetFilename, resultfilename):
    # print(str(Jobid)+str(n_tasks)+str(mode)+str(n_classes)+str(graph_conv_layers)+str(activation)+str(number_atom_features)+str(residual)+str(batchnorm)+str(dropout)+str(predictor_hidden_feats)+str(predictor_dropout)+str(self_loop)+str(currentstatus)+str(datasetFilename)+str(resultfilename))
    con = sql.connect("MOA.db")
    cur = con.cursor()
    cur.execute("Insert INTO GATModel (Jobid, ntask, mode, n_classes, att_hd,dropo,pred_hd_feat,pred_drp,alpha,self_loop,graphcn,graphat_aggmode,currentstatus, datasetFilename, resultfilename) Values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (Jobid, n_tasks, mode, n_classes, att_hd,dropo,pred_hd_feat,pred_drp,alpha,self_loop,graphcn,graphat_aggmode,currentstatus, datasetFilename, resultfilename))
    con.commit()
    con.close()
    return Jobid


def insertJobsDAG(Jobid,nclass,natom,dropo,pred_hd_feat,pred_drp,self_loop,graphcn,layersize_gather,mode,currentstatus,datasetFilename,resultfilename):
    # print(str(Jobid)+str(n_tasks)+str(mode)+str(n_classes)+str(graph_conv_layers)+str(activation)+str(number_atom_features)+str(residual)+str(batchnorm)+str(dropout)+str(predictor_hidden_feats)+str(predictor_dropout)+str(self_loop)+str(currentstatus)+str(datasetFilename)+str(resultfilename))
    con = sql.connect("MOA.db")
    cur = con.cursor()
    cur.execute("Insert INTO DAGModel (Jobid,nclass,natom,dropo,pred_hd_feat,pred_drp,self_loop,graphcn,layersize_gather,mode,currentstatus,datasetFilename,resultfilename) Values (?,?,?,?,?,?,?,?,?,?,?,?,?)", (Jobid,nclass,natom,dropo,pred_hd_feat,pred_drp,self_loop,graphcn,layersize_gather,mode,currentstatus,datasetFilename,resultfilename))
    con.commit()
    con.close()
    return Jobid

#  #       job_id,created,email,job_name,status,mode,n_tasks,n_classes,dropout,no_atom_features,max_atoms,n_graph_feat,n_outputs,layer_sizes_gather,layer_sizes,uncertainity,self_loop,learning_rate,epoch,csv_name
# Jobid,ntask,nclass,natom,bond_ft,dropo,pred_hd_feat,num_ts,graph_ft,mode
def insertJobsAttentive(Jobid,ntask,nclass,natom,bond_ft,dropo,pred_hd_feat,pred_drp,self_loop,num_ts,graph_ft,mode,currentstatus,datasetFilename,resultfilename):
    # print(str(Jobid)+str(n_tasks)+str(mode)+str(n_classes)+str(graph_conv_layers)+str(activation)+str(number_atom_features)+str(residual)+str(batchnorm)+str(dropout)+str(predictor_hidden_feats)+str(predictor_dropout)+str(self_loop)+str(currentstatus)+str(datasetFilename)+str(resultfilename))
    con = sql.connect("MOA.db")
    cur = con.cursor()
    cur.execute("Insert INTO AttentiveModel (Jobid,ntask,nclass,natom,bond_ft,dropo,pred_hd_feat,pred_drp,self_loop,num_ts,graph_ft,mode,currentstatus,datasetFilename,resultfilename) Values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (Jobid,ntask,nclass,natom,bond_ft,dropo,pred_hd_feat,pred_drp,self_loop,num_ts,graph_ft,mode,currentstatus,datasetFilename,resultfilename))
    con.commit()
    con.close()
    return Jobid

def isEmailNotPresentInsert(email,name, password):
    con = sql.connect("MOA.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE email=?", (email,))
    users = cur.fetchall()
    if users == []:
        print("not a user...")
#         insert and make a new user
        cur.execute("Insert into users (Email, Name, Password) Values (?,?,?)", (email, name, password))
        con.commit()
        con.close()
        return True
    else:
#         users present
        return False

def checkUser(email,password):
    con = sql.connect("MOA.db")
    cur = con.cursor()
    cur.execute("SELECT password FROM users WHERE email=?", (email,))
    users = cur.fetchall()
    if users == []:
        print("not a user")
        return False
    elif password == users[0][0]:
        print("enter....")
        return True
    else:
        print("Wrong pwd..")
        return False

def retrieveAllJobs(email):
    con = sql.connect("MOA.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM GCNModel where email=?",(email,))
    jobs = cur.fetchall()
    con.close()
    return jobs

def fromGCN(job_id):
    con = sql.connect("MOA.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM GCNModel where Jobid=? ", (job_id,))
    row = cur.fetchall()
    print(row)
    con.close()
    return row


def getJobs(email,mode):
    if mode == 'gcn':
        con = sql.connect("MOA.db")
        cur = con.cursor()
        cur.execute("SELECT Jobid FROM createdjobs where email=? and TypeOfModel='gcn' ", (email,))
        jobs = cur.fetchall()
        print(jobs)
        con.close()
    if mode == 'dag':
        con = sql.connect("MOA.db")
        cur = con.cursor()
        cur.execute("SELECT Jobid FROM createdjobs where email=? and TypeOfModel='dag' ", (email,))
        jobs = cur.fetchall()
        print(jobs)
        con.close()
    if mode == 'gat':
        con = sql.connect("MOA.db")
        cur = con.cursor()
        cur.execute("SELECT Jobid FROM createdjobs where email=? and TypeOfModel='gat' ", (email,))
        jobs = cur.fetchall()
        print(jobs)
        con.close()
    if mode == 'attentive':
        con = sql.connect("MOA.db")
        cur = con.cursor()
        cur.execute("SELECT Jobid FROM createdjobs where email=? and TypeOfModel='attentive' ", (email,))
        jobs = cur.fetchall()
        print(jobs)
        con.close()
    return jobs

def getCurrentStatus(jobid,mode):
    if mode == 'gcn':
        con = sql.connect("MOA.db")
        cur = con.cursor()
        cur.execute("SELECT currentstatus FROM GCNModel where Jobid=?  ", (jobid,))
        status = cur.fetchall()
        print(jobid)
        print(status[0][0])
        con.close()
        return status[0][0]
    if mode == 'gat':
        con = sql.connect("MOA.db")
        cur = con.cursor()
        cur.execute("SELECT currentstatus FROM GATModel where Jobid=?  ", (jobid,))
        status = cur.fetchall()
        print(status[0][0])
        con.close()
        return status[0][0]
    if mode == 'dag':
        con = sql.connect("MOA.db")
        cur = con.cursor()
        cur.execute("SELECT currentstatus FROM DAGModel where Jobid=?  ", (jobid,))
        status = cur.fetchall()
        print(status[0][0])
        con.close()
        return status[0][0]
    if mode == 'attentive':
        con = sql.connect("MOA.db")
        cur = con.cursor()
        cur.execute("SELECT currentstatus FROM AttentiveModel where Jobid=?  ", (jobid,))
        status = cur.fetchall()
        print(status[0][0])
        con.close()
        return status[0][0]


def updateCurrentStatus(jobid,mode,status):
    if mode == 'gcn':
        con = sql.connect("MOA.db")
        cur = con.cursor()
        cur.execute("UPDATE GCNModel SET currentstatus =? WHERE Jobid =?", (status, jobid,))
        con.commit()
        con.close()
    if mode == 'gat':
        con = sql.connect("MOA.db")
        cur = con.cursor()
        cur.execute("UPDATE GATModel SET currentstatus =? WHERE Jobid =?", (status, jobid,))
        con.commit()
        con.close()
    if mode == 'attentive':
        con = sql.connect("MOA.db")
        cur = con.cursor()
        cur.execute("UPDATE AttentiveModel SET currentstatus =? WHERE Jobid =?", (status, jobid,))
        con.commit()
        con.close()
    if mode == 'dag':
        con = sql.connect("MOA.db")
        cur = con.cursor()
        cur.execute("UPDATE DAGModel SET currentstatus =? WHERE Jobid =?", (status, jobid,))
        con.commit()
        con.close()


def get_all_job(email):
    all_jobs = []
    modes = ['gcn','dag', 'gat','attentive']
    for mode in modes:
        jobids = getJobs(email,mode)
        for job in jobids:
            currentstatus = getCurrentStatus(job[0],mode)
            all_jobs.append(Job(job[0],mode,currentstatus[0]))

    print(all_jobs)
    # for i in all_jobs:
    #     print(i.job_id)
    #     print(i.mode)
    #     print(i.status)
    return all_jobs

def insertIntoCreatedJob(Jobid,Email, Mode):
    con = sql.connect("MOA.db")
    cur = con.cursor()
    cur.execute("Insert into createdjobs (Email, Jobid, TypeOfModel) Values (?,?,?)", (Email, Jobid, Mode))
    con.commit()
    con.close()



def getMaxId():
    con = sql.connect("MOA.db")
    cur = con.cursor()
    cur.execute("SELECT MAX(Jobid) as max_items FROM createdjobs ")
    status = cur.fetchall()
    print(status[0][0])
    con.close()
    return status[0][0]


# (5,"gcn",1)
# insertIntoCreatedJob( 8 , "rahul@gmail.com" , "gcn" )
#
# def insertJobs(email,title):
#     con = sql.connect("dtabase.s3db;")
#     cur = con.cursor()
#     job_id = random.randint(1000,2000)
#     cur.execute("SELECT job_id from deep_jobs")
#     pre_ids = cur.fetchall()
#     print(pre_ids)
#     if pre_ids!=[]:
#       while job_id in pre_ids[0]:
#         job_id = random.randint(1000, 2000)
#     created = datetime.datetime.now()
#     # created =  time.time()
#     status="created"
#     mode="Classification"
#     name = "d1.csv"
#     cur.execute("INSERT INTO deep_jobs (job_id,created,email,job_name,status,mode,csv_name) VALUES (?,?,?,?,?,?,?)", (job_id,created, email,title,status,mode,name))
#     con.commit()
#     con.close()
#     return job_id
#
# def retrieveJobs(job_id):
#     print(type(job_id))
#     con = sql.connect("dtabase.s3db;")
#     cur = con.cursor()
#
#     cur.execute("SELECT * FROM deep_jobs WHERE job_id=?", (job_id,))
#     row = cur.fetchall()
#     con.close()
#     return row
#

#
#
#
# def insertusers(u_email,u_name,pwd):
#     con = sql.connect("dtabase.s3db;")
#     cur = con.cursor()
#     created = datetime.datetime.now()
#     cur.execute("INSERT INTO users (name,email,pwd,created) VALUES (?,?,?,?)", (u_name,u_email,pwd,created))
#     con.commit()
#     con.close()
#     return "Success"
#
# def checkuser(email,pwd):
#     # print(type(job_id))
#     con = sql.connect("dtabase.s3db;")
#     cur = con.cursor()
#     cur.execute("SELECT pwd FROM users WHERE email=?", (email,))
#     users = cur.fetchall()
#     con.close()
#     print(users)
#     if users==[]:
#         print("not a user...")
#         return False
#     elif pwd == users[0][0]:
#         print("enter....")
#         return True
#     else:
#         print("Wrong pwd..")
#         return False
#     # return users
#
# def hasEmail(email):
#     con = sql.connect("dtabase.s3db;")
#     cur = con.cursor()
#
#     cur.execute("SELECT email FROM users WHERE email=?", (email,))
#     users = cur.fetchall()
#     con.close()
#     print(users)
#     if(users == []):
#         return False
#     else:
#         return True
#
# getJobs("rahul20065@iiitd.ac.in","gcn")
# getCurrentStatus( 1 ,'gcn')

# get_all_job("rahul20065@iiitd.ac.in")

# row = fromGCN(1)
# print(type(row))
# print(row[0][2])
# getCurrentStatus(9,"gcn")