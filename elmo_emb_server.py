from elmo_features import *
from flask import Flask, request, jsonify

from src.utils_float_string import float_to_string

app = Flask(__name__)


@app.route("/elmo_embed_words", methods=['GET', 'POST'])
def index():
    value = request.json['text']
    # print(value)
    value = value.split('\r\n')
    # print(value)
    res = get_elmo_embs(value, mean=False)

    vectors = float_to_string(res)
    return jsonify({'result': vectors})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4004)
