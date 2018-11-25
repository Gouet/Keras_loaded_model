import os
import sys

import socket
import io

import numpy as np
from keras.models import model_from_json
from ast import literal_eval as make_tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DNAIService:
    def __init__(self, port):
        self.model = None
        self.is_running = False
        self.commands = {
            "LOAD_MODEL": self.load_model,
            "LOAD_WEIGHTS": self.load_weights,
            "PREDICT": self.predict,
            "QUIT": self.quit
        }
        self.read_lines = []
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.connect(('127.0.0.1'.encode(), port))
        self.enable_log = False
        print(self.commands.keys())

    def LogMessage(self, msg):
    	if self.enable_log:
	        with open('log_file.info', 'a') as lfile:
	            print(msg, file=lfile)

    def send_data(self, data):
        print(data)
        self.server.send(data.encode())

    def send_error(self, msg):
        self.send_data('ERROR: %s\n' % str(msg))

    def read_data(self):
        size = len(self.read_lines)

        if size > 1:
            return self.read_lines.pop(0)

        data = self.server.recv(4096).decode('utf-8').replace('\r', '')
        lines = data.split('\n')

        if len(lines) == 0 and len(self.read_lines) == 0:
            return ''

        if len(self.read_lines) > 0:
            self.read_lines[0] += lines[0]
            self.read_lines += lines[1:]
        else:
            self.read_lines += lines

        return self.read_lines.pop(0)

    def load_model(self):
        filename = self.read_data()
        with open(filename, 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.LogMessage('Model loaded: ' + filename)

    def load_weights(self):
        filename = self.read_data()
        self.model.load_weights(filename)
        self.LogMessage('Weights loaded: ' + filename)

    def get_inputs(self):
        self.LogMessage('====Get input====')
        self.LogMessage('Read row')
        row_count = int(self.read_data())
        self.LogMessage('- row ' + str(row_count))
        self.LogMessage('Read col')
        col_count = int(self.read_data())
        self.LogMessage('- col ' + str(col_count))
        self.LogMessage('Read shape')
        shape = make_tuple(self.read_data())
        self.LogMessage('- shape ' + str(shape))
        self.LogMessage('Read inputs')
        inputs = []

        for i in range(0, row_count):
            self.LogMessage('- Line %d' % i)
            row = self.read_data()
            self.LogMessage('  - Row: %s' % row)
            splitRow = row.split(',')
            real_row = [float(value) for value in splitRow]
            if len(real_row) != col_count:
                raise SyntaxError('No enough data in row %d' % i)
            inputs.append(real_row)

        return np.array(inputs).reshape(shape)

    def predict(self):
        self.LogMessage('Starting prediction')
        inputs = self.get_inputs()
        self.LogMessage('Inputs get')
        self.LogMessage('Start prediction')
        predicts = self.model.predict(np.array([inputs]), verbose=0)
        self.LogMessage('Prediction finished: ' + str(predicts))
        response = str(len(predicts)) + '\n'
        for prediction in predicts:
            plen = len(prediction)
            for index, item in enumerate(prediction):
                response += str(item) + (',' if index + 1 < plen else '\n')
        self.send_data(response)

    def quit(self):
        self.is_running = False

    def run(self):
        print('Server is running', file=sys.stderr)
        self.is_running = True
        while self.is_running:
            self.LogMessage('Getting command')
            command = self.read_data()
            if len(command) == 0:
                print('Disconnected')
                self.LogMessage('Disconnected')
                self.is_running = False
                break
            self.LogMessage('Command Get: ' + command)
            if command in self.commands.keys():
                try:
                    self.commands[command]()
                except Exception as err:
                    self.send_error('Command %s: %s' % (command, str(err)))
                    print('Failed to execute %s:' % command, err, file=sys.stderr)
                    self.LogMessage('Error detected: ' + str(err))
            else:
                print('No such command:', command, file=sys.stderr)


def main(av):
    if len(av) != 3:
        print('Invalid arguments')
        exit(1)

    port = int(av[2])
    service = DNAIService(port)
    try:
        service.run()
    except Exception as err:
        print('Program error:', err, file=sys.stderr)
        service.send_error('Program: %s' % str(err))


if __name__ == '__main__':
    main(sys.argv)
