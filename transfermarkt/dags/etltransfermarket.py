from airflow import DAG
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator
# from airflow.operators.bash import BashOperator
# from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.mongo.hooks.mongo import MongoHook
from pymongo import MongoClient
import sys
import os
from datetime import datetime, timedelta
sys.path.append('/usr/local/airflow/dags')
from dags.player import get_data, print_done, transform



def to_csv_file(dataf):
    dag_file_path = os.path.abspath(__file__)
    csv_file_path = os.path.join(os.path.dirname(dag_file_path), 'players.csv')
    dataf.to_csv(csv_file_path, index=False)

def save_to_mongodb(dataframe):

    mongo_url = ''
    database_name = 'etl'
    collection_name='TopTransfers'
    client = MongoClient(mongo_url)
    db = client[database_name]
    data_dict = dataframe.to_dict(orient='records')
    db[collection_name].insert_many(data_dict)
    client.close()


default_args = {
    'owner': 'airflow',
    'start_date': datetime.now(),
    'schedule_interval': '@daily',
    'catchup': False,
}

dag = DAG(
    'data',
    default_args=default_args,
    description='DAG for scraping and saving transfermarkt data',
)


scrape_task = PythonOperator(
    task_id='get_trans',
    python_callable=get_data,
    # provide_context=True, 
    dag=dag,
)
transform_task = PythonOperator(
    task_id='transform',
    python_callable=transform,
    op_args=[scrape_task.output],  
    # provide_context=True,
    dag=dag,
)

csv_task = PythonOperator(
    task_id='to_csv_file',
    python_callable=to_csv_file,
    op_args=[scrape_task.output],  
    # provide_context=True,
    dag=dag,
)



load_task = PythonOperator(
    task_id='save_to_mongodb',
    python_callable=save_to_mongodb,
    
    op_args=[transform_task.output],  
    # provide_context=True,
    dag=dag,
)

print_done_dag = PythonOperator(
    task_id='print_done',
    python_callable=print_done,
    # provide_context=True,
    dag=dag,
)


scrape_task >> transform_task
scrape_task >> csv_task 

transform_task >> load_task
load_task >> print_done_dag
