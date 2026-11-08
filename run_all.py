"""Execute all analytical stages after Step 1 data has been downloaded."""
import subprocess
import sys
for name in ['step2_eda','step3_portfolios','step4_optimization','step5_risk','step6_strategies','step7_machine_learning','step8_evaluation']:
    print(f'\n===== {name} =====',flush=True)
    subprocess.run([sys.executable,'-m',f'src.{name}'],check=True)
