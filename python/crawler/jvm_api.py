from glob import glob
import site
import sys
import subprocess
from py4j.java_gateway import JavaGateway

# Find a bunch of classes
jar_file = '/usr/local/share/py4j/py4j0.10.6.jar'
subprocess.check_call(['javac', '-cp', jar_file, 'JvmAPI.java'])
proc = subprocess.Popen(['java', '-cp', '{}:.'.format(jar_file), 'JvmAPI'])

gateway = JavaGateway()
random = gateway.jvm.java.util.Random()

gateway.close()
proc.kill()
