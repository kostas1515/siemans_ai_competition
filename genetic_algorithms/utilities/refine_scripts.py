import os
os.chdir("../")
dataset = 'trainingdata_a'
directory = 'checkpoints/{}/'.format(dataset)
data_path = 'data/{}'.format(dataset)
refined_dataset_path = os.path.join("vote_dataset",dataset)
synth_vote_dataset = os.path.join("synth_vote_dataset",dataset)
if not os.path.exists(synth_vote_dataset):
    os.mkdir(synth_vote_dataset)
if not os.path.exists(refined_dataset_path):
    os.mkdir(refined_dataset_path)
if not os.path.exists(os.path.join('refined_scripts',dataset)):
    os.mkdir(os.path.join('refined_scripts',dataset))

for filename in os.listdir(directory):
    if  filename.endswith(".py"):
        script = os.path.join(directory,filename)
        with open(script,'r') as script:
            lines=script.readlines()
            for k,line in enumerate(lines):
                if line.startswith("#"):
                    break
            lines[k-1] = "from sklearn.metrics import f1_score\n"
            lines[k+1] = "tpot_data = pd.read_csv('../{}.csv')\n".format(data_path)
            lines[k+2] = "features = tpot_data.drop('l_i', axis=1).drop('Unnamed: 0', axis=1)\n"
            lines[k+4] = "            train_test_split(features, tpot_data['l_i'], random_state=1)\n"
            lines.append("f1 = f1_score(testing_target,results)\n")
            lines.append("df = pd.DataFrame(columns=['x_i1', 'x_i2','l_0','l_1'])\n")
            lines.append("df['x_i1'],df['x_i2'] = testing_features['x_i1'],testing_features['x_i2']\n")
            lines.append("onehot=np.array([[f1,1-f1],[1-f1,f1]])\n")
            lines.append("df['l_0'],df['l_1'] = (onehot[results])[:,0],(onehot[results])[:,1]\n")
            lines.append(f"df.to_csv('../{refined_dataset_path}/{filename.split('.')[0]}.csv')\n")

            lines.append("sdf=pd.DataFrame(columns=['x_i1', 'x_i2','l_0','l_1'])\n")
            lines.append("np.random.seed(seed=25)\n")
            lines.append("synth_data = np.random.random_sample((1000000,2))\n")
            lines.append("synth_results = exported_pipeline.predict(synth_data)\n")
            lines.append("sdf['x_i1'],sdf['x_i2'] = synth_data[:,0],synth_data[:,1]\n")
            lines.append("sdf['l_0'],sdf['l_1'] = (onehot[synth_results])[:,0],(onehot[synth_results])[:,1]\n")
            lines.append(f"sdf.to_csv('../{synth_vote_dataset}/{filename.split('.')[0]}.csv')")
        filename = os.path.join('refined_scripts',dataset,filename)
        with open(filename,'w') as rs:
            rs.writelines(lines)