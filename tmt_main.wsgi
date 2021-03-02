#! /usr/bin/python

import sys
sys.path.insert(0, "/var/www/tmt_main")
sys.path.insert(0,'/opt/conda/lib/python3.6/site-packages')
sys.path.insert(0, "/opt/conda/bin/")

import os
os.environ['PYTHONPATH'] = '/opt/conda/bin/python'

from tmt_main import app as application