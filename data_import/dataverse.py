import requests
import json
import os
import urllib3
import cv2
import numpy as np
import threading
import logging
import re
import csv


from tqdm import tqdm
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s.%(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


class DVDataVerse:
    def __init__(self, base_url, api_token):
        self.base_url = base_url
        self.api_token = api_token

    def get_base_url(self):
        return self.base_url

    def get_api_token(self):
        return self.api_token

    def check_connection(self):
        request_url = '%s/api/info/version' % self.base_url
        resp = requests.get(request_url, verify=False)
        connection_info = json.loads(resp.content)
        logging.info('DATAVERSE: Connection to %s %s' % (self.base_url, connection_info))


class DVDataSet:
    def __init__(self, dataverse, pid=None):
        self.dv = dataverse
        self.pid = pid
        self.base_url = self.dv.get_base_url()
        self.api_token = self.dv.get_api_token()
        self.list_of_files = None
        self.query_list_of_files()

    def query_list_of_files(self):
        request_url = '%s/api/datasets/:persistentId/?persistentId=%s&key=%s' % (self.base_url, self.pid, self.api_token)
        resp = requests.get(request_url, verify=False)
        dataset_info = json.loads(resp.content)
        file_list = dataset_info['data']['latestVersion']['files']
        logging.info('DATASET: %s' % dataset_info['data']['persistentUrl'])
        self.list_of_files = [file['dataFile'] for file in file_list]

    def get_list_of_files(self):
        if self.list_of_files is None:
            self.query_list_of_files()
        return self.list_of_files

    def get_dataverse(self):
        return self.dv

    def get_pid(self):
        return self.pid


class DVFileUpload:
    def __init__(self, dataset, file_path):
        self.ds = dataset
        self.dv = self.ds.get_dataverse()
        self.base_url = self.dv.get_base_url()
        self.api_token = self.dv.get_api_token()
        self.dataset_pid = self.ds.get_pid()
        self.file_path = file_path
        _, self.file_extension = os.path.splitext(self.file_path)
        self.file_name = os.path.basename(self.file_path)

    def upload(self):
        if self.is_duplicate():
            logging.debug('File %s is duplicate, upload skipped' % self.file_name)
            return True
        else:
            return self.upload_file()


    def upload_file(self):
        add_pars = {'description': ''}
        payload = {'jsonData': json.dumps(add_pars)}
        file_extension = self.file_extension.replace('.', '')
        upload_file_url = '%s/api/datasets/:persistentId/add?persistentId=%s&key=%s' % (self.base_url, self.dataset_pid, self.api_token)
        if file_extension in ['png', 'jpg']:
            raw_frame = cv2.imread(self.file_path)
            _, frame_arr = cv2.imencode('.' + file_extension, raw_frame)
            data = frame_arr.tobytes()
            content_type = 'image/' + file_extension

        elif file_extension == 'json':
            with open(self.file_path) as json_file:
                json_data = json.load(json_file)
            data = json.dumps(json_data, indent=4)
            content_type = 'application/json'
        else:
            return False

        file_content = {'file': (self.file_name, data, content_type)}
        resp_file_upload = requests.post(upload_file_url,
                                         data=payload,
                                         files=file_content,
                                         verify=False,
                                         timeout=5)
        if resp_file_upload.ok:
            return True
        else:
            return False

    def is_duplicate(self):
        list_of_file_names = [file['filename'] for file in self.ds.get_list_of_files()]
        return self.file_name in list_of_file_names

    def get_file_name(self):
        return self.file_name


class DVFileUploader:
    def __init__(self, dv_file):
        self.dv_file = dv_file

    def upload(self):
        uploaded = False
        while not uploaded:
            uploaded = self.upload_attempt()

    def upload_attempt(self):
        try:
            upload_status = self.dv_file.upload()
            if upload_status:
                logging.debug('File %s: uploaded' % self.dv_file.get_file_name())
                return True
            else:
                logging.debug('File %s: upload attempt failed' % self.dv_file.get_file_name())
                return False
        except Exception:
            logging.debug('File %s: upload attempt failed' %self.dv_file.get_file_name())
            return False

    def get_dv_file(self):
        return self.dv_file


class DVFilesUploader:
    def __init__(self, dv_file_uploaders, no_workers=16):
        self.dv_file_uploaders = dv_file_uploaders
        self.no_workers = no_workers
        self.workers = []
        self.__initialise_workers()

    def __initialise_workers(self):
        for i in range(self.no_workers):
            self.workers.append(Worker())

    def run(self):
        files_left = len(self.dv_file_uploaders)
        logging.info('Files to upload %s' % files_left)
        progress_bar = tqdm(total=int(files_left))
        uploaders_iter = iter(self.dv_file_uploaders)

        while files_left:
            try:
                for (worker, worker_id) in zip(self.workers, range(self.no_workers)):
                    if not worker.is_busy():
                        file_uploader = next(uploaders_iter)
                        logging.debug('Worker %i uploads %s' % (worker_id, file_uploader.get_dv_file().get_file_name()))
                        worker.assign_task(file_uploader.upload, [])
                        worker.run()
                        files_left -= 1
                        progress_bar.update(1)
            except StopIteration:
                worker_completed = False
                while not worker_completed:
                    worker_completed = True
                    for worker in self.workers:
                        work_completed = not worker.is_busy()
                        work_completed *= work_completed


class DVFileDownload:
    def __init__(self, dataset, fid=None, file_name=None, content_type=None):
        self.ds = dataset
        self.fid = fid
        self.dv = self.ds.get_dataverse()
        self.base_url = self.dv.get_base_url()
        self.api_token = self.dv.get_api_token()
        self.file_name = file_name
        self.content_type = content_type

    def download_by_id(self):
        if self.fid is None:
            return False
        request_url = '%s/api/access/datafile/%s?key=%s' % (self.base_url, self.fid, self.api_token)
        return requests.get(request_url, verify=False, timeout=5)

    def get_file_name(self):
        return self.file_name

    def get_content_type(self):
        return self.content_type

    def get_file_id(self):
        return self.fid


class DVFileDownloader:
    def __init__(self, dv_file, save_path):
        self.save_path = save_path
        self.dv_file = dv_file

    def save(self):
        saved = False
        while not saved:
            saved = self.save_attempt()

    def save_attempt(self):
        try:
            if self.save_path is None:
                return False

            resp_body = self.dv_file.download_by_id()
            filename = self.dv_file.get_file_name()
            file_full_path = os.path.join(self.save_path, filename)
            content_type = self.dv_file.get_content_type()

            if content_type.find('image') != -1:
                image_string = resp_body.content
                img = cv2.imdecode(np.fromstring(image_string, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(file_full_path, img)
            elif content_type.find('application/json') != -1:
                metadata = resp_body.json()
                with open(file_full_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            elif content_type.find('text/tab-separated-values') != -1:
                tab_bytes = resp_body.content
                file_writer = open(file_full_path, 'wb')
                file_writer.write(tab_bytes)
                file_writer.close()
            else:
                tab_bytes = resp_body.content
                file_writer = open(file_full_path, 'wb')
                file_writer.write(tab_bytes)
                file_writer.close()

            print(f'File {filename} saved in {file_full_path}')
            return True
        except Exception:
            print('Retrying...')
            return False

    def get_dv_file(self):
        return self.dv_file


class DVFilesDownloader:
    def __init__(self, dv_file_savers, no_workers=16):
        self.dv_file_savers = dv_file_savers
        self.no_workers = no_workers
        self.workers = []
        self.__initialise_workers()

    def run_single_thread(self):
        for dv_file_saver in self.dv_file_savers:
            dv_file_saver.save()

    def __initialise_workers(self):
        for i in range(self.no_workers):
            self.workers.append(Worker())

    def run_multi_thread(self):
        files_left = len(self.dv_file_savers)
        logging.info('Files to download %s' % files_left)
        progress_bar = tqdm(total=int(files_left))
        savers_iter = iter(self.dv_file_savers)

        while files_left:
            try:
                for (worker, worker_id) in zip(self.workers, range(self.no_workers)):
                    if not worker.is_busy():
                        file_saver = next(savers_iter)
                        logging.debug('Worker %i downloads %s' % (worker_id, file_saver.get_dv_file().get_file_id()))
                        worker.assign_task(file_saver.save, [])
                        worker.run()
                        files_left -= 1
                        progress_bar.update(1)
            except StopIteration:
                worker_completed = False
                while not worker_completed:
                    worker_completed = True
                    for worker in self.workers:
                        work_completed = not worker.is_busy()
                        work_completed *= work_completed
                break


class Worker:
    def __init__(self):
        self.busy = False
        self.thread = None
        self.task = None
        self.args = None
        self.task_assigned = False

    def assign_task(self, task, args):
        self.task = task
        self.args = args
        self.task_assigned = True
        self.busy = True

    def run(self):
        if self.task_assigned:
            self.thread = threading.Thread(target=self.__execution)
            self.thread.start()

    def __execution(self):
        self.task(*self.args)
        self.busy = False

    def is_busy(self):
        return self.busy
