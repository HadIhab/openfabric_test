from openfabric_pysdk.starter import OpenfabricStarter
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from flask import Flask, request
import pandas as pd
import numpy as np
from main import execute


if __name__ == '__main__':

    data = pd.read_csv("SQuAD_data.csv")
    random_num = np.random.randint(0, len(data))
    print('[QUESTION]: {}'.format(data["question"][random_num]))
    print('[TEXT]: {}'.format(data["text"][random_num]))

    #OpenfabricStarter.ignite(debug=False, host="0.0.0.0", port=5000),
    app = Flask(__name__)

    @app.route("/answer", methods=["POST"])
    def answer_question():
        print('[INFO]: Preparing data..')
        data = request.get_json()
        answer = execute(data, ray=None)
        return {"answer": answer}
    app.run(debug=False, host="0.0.0.0", port=5000)
