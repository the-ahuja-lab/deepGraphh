# flask_blog/app.py
import os
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, send_file,flash
from flask import send_from_directory
import init_db as database
from Models import GCNModel as gcn
from Models import GATModel as gat
from Models import DAGModel as dag
from Models import AttentiveFPmodel as att
from Models import CrossValid as cv
from zipfile import ZipFile
import random, threading, webbrowser
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
# from django.apps import AppConfig as app_config
from pathlib import Path

UPLOAD_FOLDER = 'Dataset'

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
# Session()
# sess = Session()
# sess.init_app(app)
app.config['Dataset'] = UPLOAD_FOLDER

gl_jobid = None


def fetch_app_config_path():
    prnt_fld = Path.home()
    # print(prnt_fld)
    fld_path = os.path.join(prnt_fld, "ml_olfa", ".app_config")
    # print(fld_path)
    if not os.path.exists(fld_path):
        os.makedirs(fld_path, exist_ok=True)

    return fld_path


def fetch_all_jobs_path():
    """
    This method creates (if doesnot exists) a folder where all jobs will be stored
    :return: return all jobs folder path
    """
    prnt_fld = Path.home()
    fld_path = os.path.join(prnt_fld, "ml_olfa", "all_jobs")
    print(fld_path)
    if not os.path.exists(fld_path):
        os.makedirs(fld_path, exist_ok=True)

    return fld_path


fetch_all_jobs_path()


# fetch_app_config_path()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/reroute')
def reroute():
    return render_template('index.html')


@app.route('/start', methods=['GET', 'POST'])
def start():
    Jobid = 0
    # flash("Submit form")
    Jobid = database.getMaxId() + 1
    if request.method == 'POST':
        # flash("Form Submitted, pipeline running ...")
        Jobid = database.getMaxId() + 1

        all_jobs_fld = fetch_all_jobs_path()
        # create the respective job id folder inside all jobs folder
        job_fld = os.path.join(all_jobs_fld, str(Jobid))

        if not os.path.exists(job_fld):
            os.makedirs(job_fld, exist_ok=True)
        else:
            while os.path.exists(job_fld):
                Jobid = Jobid + 1
                job_fld = os.path.join(all_jobs_fld, str(Jobid))
            os.makedirs(job_fld, exist_ok=True)
        if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
            print(request.environ['REMOTE_ADDR'])
            Email = request.environ['REMOTE_ADDR']
        else:
            print(request.environ['HTTP_X_FORWARDED_FOR'])
            Email = request.environ['HTTP_X_FORWARDED_FOR']
        print("submit registerform")
        print(session.get("name"))
        user_form = request.form
        user_form_json = user_form.to_dict()

        print("Here", user_form_json)
        # Email = session["name"]
        # Email = 'rahul20065@iiitd.ac.in'
        try:
            ntask = user_form_json['ntask']
            nclass = user_form_json['nclass']
            natom = user_form_json['natom']
            dropo = user_form_json['dropo']
            pred_hd_feat = user_form_json['pred_hd_feat']
            pred_drp = user_form_json['pred_drp']
            batchnorm = user_form_json['norm']
            if (batchnorm == 'on'):
                batchnorm = True
            else:
                batchnorm = False
            self_loop = user_form_json['self_loop']
            if (self_loop == 'on'):
                self_loop = True
            else:
                self_loop = False
            residual = user_form_json['residual']
            if (residual == 'on'):
                residual = True
            else:
                residual = False
            graphcn = user_form_json['graphcn']
            mode = user_form_json['mode']
            activation_fxn = user_form_json['factiv']
            print(activation_fxn)
            if (mode == 'on'):
                mode = "Classification"
            else:
                mode = "Regression"

            if 'CrossValidation' in user_form_json:
                crossValid = user_form_json['CrossValidation']
                flash("Processing For each folds")
                if (crossValid == 'on'):
                    crossValid = "5"
                elif (crossValid == 'off'):
                    crossValid = "10"
            else:
                crossValid = 'None'

            file = request.files['csvfl']
            test_file = request.files['test_csv']
            # Jobid = database.getMaxId() + 1
            print(str(ntask) + str(nclass) + str(natom) + str(dropo) + str(pred_hd_feat) + str(self_loop) + str(
                residual) + str(graphcn))

            all_jobs_fld = fetch_all_jobs_path()
            # create the respective job id folder inside all jobs folder
            job_fld = os.path.join(all_jobs_fld, str(Jobid))
            os.makedirs(job_fld, exist_ok=True)
            data_fld_path = os.path.join(job_fld, "result")
            os.makedirs(data_fld_path, exist_ok=True)
            Model_Parameters_path = os.path.join(data_fld_path, "Model_Parameters.txt")

            with open(Model_Parameters_path, 'w') as f:
                for key, value in user_form_json.items():
                    f.write('%s:%s\n' % (key, value))
        finally:
            print(" == ")

        database.insertJobs(Jobid, ntask, mode, nclass, graphcn, "activation", natom, residual, batchnorm, dropo,
                            pred_hd_feat, pred_drp, self_loop, "1", file.filename, "result filename")
        database.insertIntoCreatedJob(Jobid, Email, "gcn")
        gl_jobid = Jobid
        print("start job inserted gl_jobid " + str(gl_jobid))
        filename = secure_filename(file.filename)
        test_filename = secure_filename(test_file.filename)
        print('test_filename')
        print(test_filename)
        # config for this jobid
        all_jobs_fld = fetch_all_jobs_path()
        # create the respective job id folder inside all jobs folder
        job_fld = os.path.join(all_jobs_fld, str(Jobid))
        os.makedirs(job_fld, exist_ok=True)
        data_fld_path = os.path.join(job_fld, "data")
        os.makedirs(data_fld_path, exist_ok=True)
        file.save(os.path.join(data_fld_path, "user_data.csv"))
        test_file.save(os.path.join(data_fld_path, "test_data.csv"))

        # file.save(os.path.join(app.config['Dataset'], filename))
        # test_file.save(os.path.join(app.config['Dataset'], test_filename))
        print(filename)
        if crossValid != 'None':
            cv.applyCV('gcn', crossValid, Jobid)
            flash("Cross Validation Result Downloading ...")
            return redirect(url_for('downres', variable= Jobid))

        else:
            model = gcn.GcnModel(Jobid, "created", "gmail.com", "gcn", "1", "classification", residual, "batchnorm",
                                 "activation",
                                 "64,64", dropo, "pred_dropout", natom, pred_hd_feat, "hidden_feature",
                                 ntask, self_loop, "learning_rate", "epoch", filename)
            model.fit_predict()
            model.results()
            # flash("Results Downloading ...")
            # return redirect(url_for('downres', variable= Jobid))


        
    return render_template('start.html', jobid=Jobid)


@app.route('/attnstart', methods=['GET', 'POST'])
def attnstart():
    Jobid = 0
    Jobid = database.getMaxId() + 1
    if request.method == 'POST':
        Jobid = database.getMaxId() + 1

        all_jobs_fld = fetch_all_jobs_path()
        # create the respective job id folder inside all jobs folder
        job_fld = os.path.join(all_jobs_fld, str(Jobid))

        if not os.path.exists(job_fld):
            os.makedirs(job_fld, exist_ok=True)
        else:
            while os.path.exists(job_fld):
                Jobid = Jobid + 1
                job_fld = os.path.join(all_jobs_fld, str(Jobid))
            os.makedirs(job_fld, exist_ok=True)
        if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
            print(request.environ['REMOTE_ADDR'])
            Email = request.environ['REMOTE_ADDR']
        else:
            print(request.environ['HTTP_X_FORWARDED_FOR'])
            Email = request.environ['HTTP_X_FORWARDED_FOR']

        print("submit registerform")
        print(session.get("name"))
        user_form = request.form
        user_form_json = user_form.to_dict()

        print("Here", user_form_json)
        # Here {'ntask': '2', 'nclass': '2', 'natom': '30', 'bond_ft': '11', 'dropo': '0', 'pred_hd_feat': '3', 'num_ts': '2', 'graph_ft': '200',
        # 'self_loop': 'on', 'mode': 'on'}
        # Email = 'rahul20065@iiitd.ac.in'
        try:
            ntask = user_form_json['ntask']
            nclass = user_form_json['nclass']
            natom = user_form_json['natom']
            bond_ft = user_form_json['bond_ft']
            dropo = user_form_json['dropo']
            pred_hd_feat = user_form_json['pred_hd_feat']
            num_ts = user_form_json['num_ts']
            graph_ft = user_form_json['graph_ft']
            mode = user_form_json['mode']
            self_loop = user_form_json['self_loop']
            if (self_loop == 'on'):
                self_loop = True
            else:
                self_loop = False
            if (mode == 'on'):
                mode = "Classification"
            else:
                mode = "Regression"

            if 'CrossValidation' in user_form_json:
                crossValid = user_form_json['CrossValidation']
                if (crossValid == 'on'):
                    crossValid = "5"
                elif (crossValid == 'off'):
                    crossValid = "10"
            else:
                crossValid = 'None'
            file = request.files['csvfl']
            test_file = request.files['test_csv']
            # Jobid = database.getMaxId() + 1
            #         print(str(ntask)+str(nclass)+str(natom)+str(dropo)+str(pred_hd_feat)+str(self_loop)+str(residual)+str(graphcn))
            all_jobs_fld = fetch_all_jobs_path()
            # create the respective job id folder inside all jobs folder
            job_fld = os.path.join(all_jobs_fld, str(Jobid))
            os.makedirs(job_fld, exist_ok=True)
            data_fld_path = os.path.join(job_fld, "result")
            os.makedirs(data_fld_path, exist_ok=True)
            Model_Parameters_path = os.path.join(data_fld_path, "Model_Parameters.txt")

            with open(Model_Parameters_path, 'w') as f:
                for key, value in user_form_json.items():
                    f.write('%s:%s\n' % (key, value))
        finally:
            print(" == ")
            # Jobid,ntask,nclass,natom,bond_ft,dropo,pred_hd_feat,pred_drp,self_loop,num_ts,graph_ft,mode,currentstatus,datasetFilename,resultfilename
        database.insertJobsAttentive(Jobid, ntask, nclass, natom, bond_ft, dropo, pred_hd_feat, "pred_drp", self_loop,
                                     num_ts, graph_ft, mode, "1", "user_train.csv", "resultfilename")
        database.insertIntoCreatedJob(Jobid, Email, "attentive")
        gl_jobid = Jobid
        print("start job inserted gl_jobid " + str(gl_jobid))
        filename = secure_filename(file.filename)
        test_filename = secure_filename(test_file.filename)
        print('test_filename')
        print(test_filename)
        # config for this jobid
        all_jobs_fld = fetch_all_jobs_path()
        # create the respective job id folder inside all jobs folder
        job_fld = os.path.join(all_jobs_fld, str(Jobid))
        os.makedirs(job_fld, exist_ok=True)
        data_fld_path = os.path.join(job_fld, "data")
        os.makedirs(data_fld_path, exist_ok=True)
        file.save(os.path.join(data_fld_path, "user_data.csv"))
        test_file.save(os.path.join(data_fld_path, "test_data.csv"))
        # job_id,created,email,          job_name,status,mode,n_tasks,                  n_classes,dropout,        no_atom_features,max_atoms,       n_graph_feat,n_outputs,                 layer_sizes_gather,layer_sizes,uncertainity,self_loop,learning_rate,epoch,csv_name
        if crossValid != 'None':
            cv.applyCV('attentiveFP', crossValid, Jobid)
            flash("Cross Validation Result Downloading ...")
            return redirect(url_for('downres', variable= Jobid))

        else:
            model = att.AttentiveFPmodel(Jobid, "created", Email, "attentive", "1", mode=mode, n_tasks=ntask,
                                         n_classes=nclass, dropout=dropo, no_atom_features=natom, max_atoms=None,
                                         n_graph_feat=graph_ft, n_outputs=None, layer_sizes_gather=None, layer_sizes=None,
                                         uncertainity=None, self_loop=self_loop, learning_rate=0.0001, epoch=100,
                                         csv_name="user_data.csv")
            model.fit_predict()
            model.results()
        
    return render_template('attnstart.html', jobid=Jobid)


@app.route('/graphattnstart', methods=['GET', 'POST'])
def graphattnstart():
    Jobid = 0
    Jobid = database.getMaxId() + 1
    if request.method == 'POST':
        Jobid = database.getMaxId() + 1

        all_jobs_fld = fetch_all_jobs_path()
        # create the respective job id folder inside all jobs folder
        job_fld = os.path.join(all_jobs_fld, str(Jobid))

        if not os.path.exists(job_fld):
            os.makedirs(job_fld, exist_ok=True)
        else:
            while os.path.exists(job_fld):
                Jobid = Jobid + 1
                job_fld = os.path.join(all_jobs_fld, str(Jobid))
            os.makedirs(job_fld, exist_ok=True)
        if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
            print(request.environ['REMOTE_ADDR'])
            Email = request.environ['REMOTE_ADDR']
        else:
            print(request.environ['HTTP_X_FORWARDED_FOR'])
            Email = request.environ['HTTP_X_FORWARDED_FOR']

        print("submit registerform")
        print(session.get("name"))
        user_form = request.form
        user_form_json = user_form.to_dict()

        print("Here", user_form_json)
        # Email = 'rahul20065@iiitd.ac.in'
        try:
            ntask = user_form_json['ntask']
            nclass = user_form_json['nclass']
            att_hd = user_form_json['att_hd']
            natom = user_form_json['natom']
            dropo = user_form_json['dropo']
            pred_hd_feat = user_form_json['pred_hd_feat']
            pred_drp = user_form_json['pred_drp']
            alpha = user_form_json['alpha']
            graphcn = user_form_json['graphcn']
            graphat_aggmode = user_form_json['graphat_aggmode']
            mode = user_form_json['mode']
            self_loop = user_form_json['self_loop']
            if (self_loop == 'on'):
                self_loop = True
            else:
                self_loop = False
            if (mode == 'on'):
                mode = "classification"
            else:
                mode = "regression"

            if 'CrossValidation' in user_form_json:
                crossValid = user_form_json['CrossValidation']
                if (crossValid == 'on'):
                    crossValid = "5"
                elif (crossValid == 'off'):
                    crossValid = "10"
            else:
                crossValid = 'None'
                
            file = request.files['csvfl']
            test_file = request.files['test_csv']

            all_jobs_fld = fetch_all_jobs_path()
            # create the respective job id folder inside all jobs folder
            job_fld = os.path.join(all_jobs_fld, str(Jobid))
            os.makedirs(job_fld, exist_ok=True)
            data_fld_path = os.path.join(job_fld, "result")
            os.makedirs(data_fld_path, exist_ok=True)
            Model_Parameters_path = os.path.join(data_fld_path, "Model_Parameters.txt")

            with open(Model_Parameters_path, 'w') as f:
                for key, value in user_form_json.items():
                    f.write('%s:%s\n' % (key, value))

            # Jobid = database.getMaxId() + 1
            # print(str(ntask)+str(nclass)+str(natom)+str(dropo)+str(pred_hd_feat)+str(self_loop)+str(residual)+str(graphcn))
        finally:
            print(" == ")
        database.insertJobsGAT(Jobid, ntask, mode, nclass, att_hd, dropo, pred_hd_feat, pred_drp, alpha, self_loop,
                               graphcn, graphat_aggmode, "1", file.filename, "result filename")
        database.insertIntoCreatedJob(Jobid, Email, "gat")
        gl_jobid = Jobid
        print("start job inserted gl_jobid " + str(gl_jobid))
        filename = secure_filename(file.filename)
        test_filename = secure_filename(test_file.filename)
        print('test_filename')
        print(test_filename)
        # config for this jobid
        all_jobs_fld = fetch_all_jobs_path()
        # create the respective job id folder inside all jobs folder
        job_fld = os.path.join(all_jobs_fld, str(Jobid))
        os.makedirs(job_fld, exist_ok=True)
        data_fld_path = os.path.join(job_fld, "data")
        os.makedirs(data_fld_path, exist_ok=True)
        file.save(os.path.join(data_fld_path, "user_data.csv"))
        test_file.save(os.path.join(data_fld_path, "test_data.csv"))
        print(filename)
        # job_id, created, email, job_name, status, mode, residual, n_tasks, activation, n_classes, dropout, pred_dropout, no_atom_features, graph_attention_layers, hidden_feature, alpha, self_loop, n_attention_heads, agg_modes, learning_rate, epoch, csv_name):
        if crossValid != 'None':
            cv.applyCV('gat', crossValid, Jobid)
            flash("Cross Validation Result Downloading ...")
            return redirect(url_for('downres', variable= Jobid))

        else:
            model = gat.GatModel(job_id=Jobid, created="created", email=Email, job_name="gat", status="1", mode=mode,
                                 residual="true", n_tasks=ntask, activation="activation", n_classes=nclass, dropout=dropo,
                                 pred_dropout=pred_drp, no_atom_features="2", graph_attention_layers=graphcn,
                                 hidden_feature=None, alpha=alpha, self_loop=self_loop, n_attention_heads=att_hd,
                                 agg_modes=graphat_aggmode, learning_rate=0.001, epoch=100, csv_name="csv_name")
            model.fit_predict()
            model.results()
        
    return render_template('graphattnstart.html', jobid=Jobid)


@app.route('/dagstart', methods=['GET', 'POST'])
def dagstart():
    Jobid = 0
    Jobid = database.getMaxId() + 1
    if request.method == 'POST':
        Jobid = database.getMaxId() + 1

        all_jobs_fld = fetch_all_jobs_path()
        # create the respective job id folder inside all jobs folder
        job_fld = os.path.join(all_jobs_fld, str(Jobid))

        if not os.path.exists(job_fld):
            os.makedirs(job_fld, exist_ok=True)
        else:
            while os.path.exists(job_fld):
                Jobid = Jobid + 1
                job_fld = os.path.join(all_jobs_fld, str(Jobid))
            os.makedirs(job_fld, exist_ok=True)
        if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
            print(request.environ['REMOTE_ADDR'])
            Email = request.environ['REMOTE_ADDR']
        else:
            print(request.environ['HTTP_X_FORWARDED_FOR'])
            Email = request.environ['HTTP_X_FORWARDED_FOR']

        print("submit registerform")
        print(session.get("name"))
        user_form = request.form
        user_form_json = user_form.to_dict()

        print("Here Dag model ", user_form_json)
        # Email = 'rahul20065@iiitd.ac.in'
        try:
            ntask = user_form_json['ntask']
            nclass = user_form_json['nclass']
            natom = user_form_json['natom']
            dropo = user_form_json['dropo']
            pred_hd_feat = user_form_json['pred_hd_feat']
            pred_drp = user_form_json['pred_drp']
            self_loop = user_form_json['self_loop']
            if (self_loop == 'on'):
                self_loop = True
            else:
                self_loop = False
            graphcn = user_form_json['graphcn']
            mode = user_form_json['mode']
            if (mode == 'on'):
                mode = "Classification"
            else:
                mode = "Regression"

            if 'CrossValidation' in user_form_json:
                crossValid = user_form_json['CrossValidation']
                if (crossValid == 'on'):
                    crossValid = "5"
                elif (crossValid == 'off'):
                    crossValid = "10"
            else:
                crossValid = 'None'
                
            layersize_gather = user_form_json['layersize_gather']
            file = request.files['csvfl']
            test_file = request.files['test_csv']

            all_jobs_fld = fetch_all_jobs_path()
            # create the respective job id folder inside all jobs folder
            job_fld = os.path.join(all_jobs_fld, str(Jobid))
            os.makedirs(job_fld, exist_ok=True)
            data_fld_path = os.path.join(job_fld, "result")
            os.makedirs(data_fld_path, exist_ok=True)
            Model_Parameters_path = os.path.join(data_fld_path, "Model_Parameters.txt")

            with open(Model_Parameters_path, 'w') as f:
                for key, value in user_form_json.items():
                    f.write('%s:%s\n' % (key, value))

        finally:
            print(" == ")

        database.insertJobsDAG(Jobid, nclass, natom, dropo, pred_hd_feat, pred_drp, self_loop, graphcn,
                               layersize_gather, mode, "1", file.filename, "resultfilename")
        database.insertIntoCreatedJob(Jobid, Email, "dag")
        gl_jobid = Jobid
        print("start job inserted gl_jobid " + str(gl_jobid))
        filename = secure_filename(file.filename)
        test_filename = secure_filename(test_file.filename)
        print('test_filename')
        print(test_filename)
        # config for this jobid
        all_jobs_fld = fetch_all_jobs_path()
        # create the respective job id folder inside all jobs folder
        job_fld = os.path.join(all_jobs_fld, str(Jobid))
        os.makedirs(job_fld, exist_ok=True)
        data_fld_path = os.path.join(job_fld, "data")
        os.makedirs(data_fld_path, exist_ok=True)
        file.save(os.path.join(data_fld_path, "user_data.csv"))
        test_file.save(os.path.join(data_fld_path, "test_data.csv"))
        print(filename)
        if crossValid != 'None':
            cv.applyCV('dag', crossValid, Jobid)
            flash("Cross Validation Result Downloading ...")
            return redirect(url_for('downres', variable= Jobid))

        else:
            model = dag.DagModel(job_id=Jobid, created="created", email=Email, job_name="dag", status="1", mode=mode,
                                 n_tasks=ntask, n_classes=nclass, dropout=dropo, no_atom_features=natom, max_atoms=natom,
                                 n_graph_feat=pred_hd_feat, n_outputs=None, layer_sizes_gather=layersize_gather,
                                 layer_sizes=None, uncertainity=None, self_loop=self_loop, learning_rate=0.001, epoch=100,
                                 csv_name="user_data.csv")
            model.fit_predict()
            model.results()
        
    return render_template('dagstart.html', jobid=Jobid)


@app.route('/_stuff/<jsdata>')
def stuff(jsdata):
    print(jsdata)
    print(type(jsdata))
    x = 1
    # if jsdata != None:
    print("jsdata" + str(jsdata))
    if jsdata != '0':
        x = database.getCurrentStatus(jsdata, "gcn")
    #     form is submitted then get current status wrt jobid
    return jsonify(result=x)


@app.route('/_stuffdag/<jsdata>')
def stuffdag(jsdata):
    print(jsdata)
    print(type(jsdata))
    x = 1
    # if jsdata != None:
    print("jsdata" + str(jsdata))
    if jsdata != '0':
        x = database.getCurrentStatus(jsdata, "dag")
    #     form is submitted then get current status wrt jobid
    return jsonify(result=x)


@app.route('/_stuffgat/<jsdata>')
def stuffgat(jsdata):
    print(jsdata)
    print(type(jsdata))
    x = 1
    # if jsdata != None:
    print("jsdata" + str(jsdata))
    if jsdata != '0':
        x = database.getCurrentStatus(jsdata, "gat")
    #     form is submitted then get current status wrt jobid
    print(x)
    return jsonify(result=x)


@app.route('/_stuffAttentive/<jsdata>')
def stuffAttentive(jsdata):
    print(jsdata)
    print(type(jsdata))
    x = 1
    # if jsdata != None:
    print("jsdata" + str(jsdata))
    if jsdata != '0':
        x = database.getCurrentStatus(jsdata, "attentive")
        print("current status recieved from attentive is ")
        print(x)
    #     form is submitted then get current status wrt jobid
    return jsonify(result=x)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/tutorial')
def tut():
    return render_template('tut.html')


@app.route('/datasets')
def dataset():
    return render_template('dataset.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        print("submit contact form")
        user_form = request.form
        user_form_json = user_form.to_dict()
        print("contact", user_form_json)
        sender = "odorify.ahujalab@iiitd.ac.in"
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = sender
        msg['Subject'] = user_form_json['Title']
        message = user_form_json['Message']
        msg.attach(MIMEText(message, 'plain'))
        text = msg.as_string()
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login(sender, "odorify123")
        s.sendmail(sender, sender, text)
        s.quit()
    return render_template('contact.html')


@app.route('/result')
def result():
    print(session.get("name"))
    # email = session.get("name")
    if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
        print(request.environ['REMOTE_ADDR'])
        email = request.environ['REMOTE_ADDR']
    else:
        print(request.environ['HTTP_X_FORWARDED_FOR'])
        email = request.environ['HTTP_X_FORWARDED_FOR']
    all_jobs = database.get_all_job(email)
    print("all_job_ids")
    print(all_jobs)
    if all_jobs != []:
        print("recieved")
        for i in all_jobs:
            print(i.job_id)
            print(i.mode)
            print(i.status)
    else:
        print("not recieved")
    return render_template('result.html', title="view database", all_jobs=all_jobs)


@app.route('/sample')
def sample():
    pathroot = app.root_path
    path = "/TrainSample.csv"
    finalpath = pathroot + path
    return send_file(finalpath, as_attachment=True)

@app.route('/sampleQuery')
def sampleQuery():
    pathroot = app.root_path
    path = "/QuerySample.csv"
    finalpath = pathroot + path
    return send_file(finalpath, as_attachment=True)



@app.route('/downDataset/<variable>')
def downDataset(variable):

    pathroot = app.root_path
    if variable == "1":
        path = "/Dataset/Consensus estrogen receptor alpha agonists (qualitative).zip"
        finalpath = pathroot + path
    if variable == "2":
        path = "/Dataset/Consensus Hematotoxicity.zip"
        finalpath = pathroot + path
    if variable == "3":
        path = "/Dataset/Drug-Induced Rhabdomyolysis.zip"
        finalpath = pathroot + path
    return send_file(finalpath, as_attachment=True)


@app.route('/download/<variable>')
def downres(variable):
    
    prnt_fld = Path.home()
    fld_path = os.path.join(prnt_fld, "ml_olfa", "all_jobs", str(variable), "result", "Input_data_cross_validation.csv")
    print(fld_path)
    if not os.path.exists(fld_path):
        print("Cross Validation path doesnt exist")
        all_jobs_fld = fetch_all_jobs_path()
        # create the respective job id folder inside all jobs folder
        job_fld = os.path.join(all_jobs_fld, str(variable))
        os.makedirs(job_fld, exist_ok=True)
        data_fld_path = os.path.join(job_fld, "result")
        os.makedirs(data_fld_path, exist_ok=True)
        Results_file = os.path.join(data_fld_path, "Results_file.zip")
        zipObj = ZipFile(Results_file, 'w')
        # Add multiple files to the zip
        Results = os.path.join(data_fld_path, "Input_data_test_AUC_ROC.png")
        zipObj.write(Results)
        test_r = os.path.join(data_fld_path, "Input_data_test_confusion_matrix.png")
        zipObj.write(test_r)
        train_r = os.path.join(data_fld_path, "Input_data_test_confusion_report.jpg")
        zipObj.write(train_r)
        clf_report_raw = os.path.join(data_fld_path, "Input_data_test_confusion_report_raw.csv")
        zipObj.write(clf_report_raw)
        test_ROCplot = os.path.join(data_fld_path, "Query_data_prob_matrix.csv")
        zipObj.write(test_ROCplot)
        Model_Parameters_path = os.path.join(data_fld_path, "Model_Parameters.txt")
        zipObj.write(Model_Parameters_path)
        zipObj.close()
        
    else:
        print("user cv data path exist")
        all_jobs_fld = fetch_all_jobs_path()
        # create the respective job id folder inside all jobs folder
        job_fld = os.path.join(all_jobs_fld, str(variable))
        os.makedirs(job_fld, exist_ok=True)
        data_fld_path = os.path.join(job_fld, "result")
        os.makedirs(data_fld_path, exist_ok=True)
        Results_file = os.path.join(data_fld_path, "Results_file.zip")
        zipObj = ZipFile(Results_file, 'w')
        input_data_cv_path = os.path.join(data_fld_path, "Input_data_cross_validation.csv")
        zipObj.write(input_data_cv_path)
        zipObj.close()
    doc = 'Results_file.zip'
    return send_file(Results_file, as_attachment=True)


if __name__ == "__main__":

    # port = 5000 + random.randint(0, 999)
    port = 5000
    url = "http://127.0.0.1:{0}".format(port)

    # to open default browser automatically
    if 'WERKZEUG_RUN_MAIN' not in os.environ:
        threading.Timer(1.25, lambda: webbrowser.open(url)).start()

    print("Starting application at url {}".format(url))
    # app.run(host="0.0.0.0", port=port, debug=False)
    app.run(host='0.0.0.0')
    # app.run()