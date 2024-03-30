Root=/home/user/EvoCodeBench
Data_Path=$Root/data.jsonl
Source_Code_Root=$Root/Source_Code

# Setup the execution environment for contrastors
cd $Root/Source_Code/contrastors
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for EasyVolcap
cd $Root/Source_Code/EasyVolcap
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for microagents
cd $Root/Source_Code/microagents
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for XAgent
cd $Root/Source_Code/XAgent
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for litdata
cd $Root/Source_Code/litdata
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
pip install -r requirements/test.txt
pip install -r requirements/extras.txt
pip install -r requirements/docs.txt
deactivate


# Setup the execution environment for gaussian-splatting-lightning
cd $Root/Source_Code/gaussian-splatting-lightning
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for open-iris
cd $Root/Source_Code/open-iris
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements/base.txt
# pip install -r requirements/orb.txt
pip install -r requirements/dev.txt
pip install -r requirements/server.txt
deactivate


# Setup the execution environment for tanuki_py
cd $Root/Source_Code/tanuki_py
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for skfolio
cd $Root/Source_Code/skfolio
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for UniRef
cd $Root/Source_Code/UniRef
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for scepter
cd $Root/Source_Code/scepter
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
pip install -r requirements/framework.txt
pip install -r requirements/tests.txt
pip install -r requirements/recommended.txt
pip install -r requirements/scepter_studio.txt
deactivate


# Setup the execution environment for microsearch
cd $Root/Source_Code/microsearch
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for UHGEval
cd $Root/Source_Code/UHGEval
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for Test-Agent
cd $Root/Source_Code/Test-Agent
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for Generalizable-BEV
cd $Root/Source_Code/Generalizable-BEV
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements/readthedocs.txt
pip install -r requirements/build.txt
pip install -r requirements/tests.txt
pip install -r requirements/docs.txt
pip install -r requirements/optional.txt
pip install -r requirements/runtime.txt
pip install -r requirements/mminstall.txt
deactivate


# Setup the execution environment for ollama-python
cd $Root/Source_Code/ollama-python
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for Python-Type-Challenges
cd $Root/Source_Code/Python-Type-Challenges
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for stable-fast
cd $Root/Source_Code/stable-fast
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for AutoRAG
cd $Root/Source_Code/AutoRAG
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for stable-diffusion-webui-forge
cd $Root/Source_Code/stable-diffusion-webui-forge
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for openlogprobs
cd $Root/Source_Code/openlogprobs
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for camp_zipnerf
cd $Root/Source_Code/camp_zipnerf
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for nlm-ingestor
cd $Root/Source_Code/nlm-ingestor
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for searcharray
cd $Root/Source_Code/searcharray
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


# Setup the execution environment for deluder
cd $Root/Source_Code/deluder
python -m venv myenv && source myenv/bin/activate
pip install pytest pytest-runner
pip install -r requirements.txt
deactivate


cd $Root
python update_test_path.py \ 
    --data_path $Data_Path \
    --source_code_root $Source_Code_Root