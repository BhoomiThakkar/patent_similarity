import pathlib

# Model Training related
SEED=1013  # to reproduce models
alpha=0.05  # learning rate used for doc2vec training
vec_size=100  # changed since smaller text ..
max_epochs=20  # (20-50) works ..

# Data specifics
lag=1
ENDYEAR=2020
STARTYEAR=1976

# Directory configurations
project_dir='/Volumes/Elements/PatentScore-Dec27/'
p = pathlib.Path(project_dir)
if not p.is_dir():
    p.mkdir(parents=True)

source_dir=project_dir+'RawData/'
p = pathlib.Path(source_dir)
if not p.is_dir():
    p.mkdir(parents=True)

data_dir=project_dir+'Data/'
p = pathlib.Path(data_dir)
if not p.is_dir():
    p.mkdir(parents=True)

stats_dir=project_dir+'CPC_Stats/'
p = pathlib.Path(stats_dir)
if not p.is_dir():
    p.mkdir(parents=True)

model_dir=project_dir+'Models/'
p = pathlib.Path(model_dir)
if not p.is_dir():
    p.mkdir(parents=True)

score_dir=project_dir+'Scores_Lag' + str(lag) + '/'
p = pathlib.Path(score_dir)
if not p.is_dir():
    p.mkdir(parents=True)

compiled_scores_year = project_dir + 'ConcatenatedScores_' + str(lag) + '/'
p = pathlib.Path(compiled_scores_year)
if not p.is_dir():
    p.mkdir(parents=True)

compiled_scores = project_dir + 'Scores/'
p = pathlib.Path(compiled_scores)
if not p.is_dir():
    p.mkdir(parents=True)
