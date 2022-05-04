from dataverse import DVDataVerse, DVDataSet, DVFileDownload, DVFileDownloader, DVFilesDownloader

BASE_URL = 'https://keen.zih.tu-dresden.de/'
API_TOKEN = '1face360-8275-4a7a-bce5-6a24c6bd8c24'
DATASET_PID = 'doi%3A10.5072%2FFK2%2FUX6QXW'


def download_data():
    dv = DVDataVerse(BASE_URL, API_TOKEN)
    dv.check_connection()
    ds = DVDataSet(dv, DATASET_PID)

    file_list = ds.get_list_of_files()
    files_to_download = []
    file_savers = []

    for file in file_list:
        file_id = file['id']
        file_name = file['filename']
        content_type = file['contentType']
        dv_file = DVFileDownload(ds, file_id, file_name, content_type)
        files_to_download.append(dv_file)
        file_savers.append(DVFileDownloader(dv_file, './data'))

    dv_downloader = DVFilesDownloader(file_savers, no_workers=32)
    dv_downloader.run_multi_thread()


if __name__ == '__main__':
    download_data()
