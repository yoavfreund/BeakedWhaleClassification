import pickle
import yaml

config_file = 'settings/settings_detector_xwav_default.yaml'
FILE_PATH = 'Arctic_C2_10_150728_020000.pkl'

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
config = load_config(config_file)

# load from config or preset
# config = {}
# config['saveForTPWS']

with open(FILE_PATH, 'rb') as f:
    data = pickle.load(f)
if config['saveForTPWS']:
    clickTimes, ppSignal, f, hdr, specClickTf, yFiltBuff, params = data
else:
    clickTimes, ppSignal, durClick, f, hdr,\
        nDur, deltaEnv, bw3db, yFilt, specClickTf, peakFr, yFiltBuff, params = data

print('clickTimes[:5]\n', clickTimes[:5])
print('ppSignal[:5]\n', ppSignal[:5])
print('ppSignal[:20]\n', f[:20])
print('hdr\n', hdr)
print('specClickTf[:1]\n', specClickTf[:1])
print('yFiltBuff[:1]', yFiltBuff[:1])
print('params', params)