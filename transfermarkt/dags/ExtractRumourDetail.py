from airflow import DAG
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator
from airflow.providers.mongo.hooks.mongo import MongoHook
from pymongo import MongoClient
import sys
import os
from datetime import datetime, timedelta
sys.path.append('/usr/local/airflow/dags')
from dags.RumourDetail import get_data, print_done



def to_csv_file(dataf):
    dag_file_path = os.path.abspath(__file__)
    csv_file_path = os.path.join(os.path.dirname(dag_file_path), 'rumours_detail.csv')
    dataf.to_csv(csv_file_path, index=False)


def save_to_mongodb(dataframe):

    mongo_url = 'mongodb+srv://[password]@cluster0.z9trbzx.mongodb.net/etl?retryWrites=true&w=majority&appName=Cluster0'
    database_name = 'etl'
    collection_name='RumoursDetail'
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
    'Rumour_Detail_Data',
    default_args=default_args,
    description='DAG for scraping and saving rumour data from transfermarkt.com',
)


scrape_task = PythonOperator(
    task_id='rumour_detail_data',
    python_callable=get_data,
    # provide_context=True, 
    dag=dag,
)

csv_task = PythonOperator(
    task_id='save_to_csv_file',
    python_callable=to_csv_file,
    op_args=[scrape_task.output],  
    # provide_context=True,
    dag=dag,
)



load_task = PythonOperator(
    task_id='save_to_mongodb',
    python_callable=save_to_mongodb,
    op_args=[scrape_task.output],  
    # provide_context=True,
    dag=dag,
)

print_done_dag = PythonOperator(
    task_id='print_done',
    python_callable=print_done,
    # provide_context=True,
    dag=dag,
)



scrape_task >> csv_task 

scrape_task >> load_task
load_task >> print_done_dag
