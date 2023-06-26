
# belajar di google tentang "get file from google drive django pydrive" , "Google API"
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


class Command(BaseCommand):

    help = "Load file from google drive, and save to ExtractedRegulation"

    @transaction.atomic()
    def handle(self, *args, **options):
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
        drive = GoogleDrive(gauth)
        file_added = 0
        file_not_added = 0
        count = 1

        fileList = drive.ListFile({'q': "'1B9mIJdRKXtpN3cgj8_OUDpEZn-fifob0' in parents and trashed=false"}).GetList()
        # ID for json external: 1J4WSrM70XSf4Es7LS0NpG5kcQMAVRBRf
        # ID for json internal: 1ShBf55d1ypBW3KlXocon_Ypw_de-cJjZ
        # ID for pdf external: 1B9mIJdRKXtpN3cgj8_OUDpEZn-fifob0
        # ID for pdf internal: 1EK5b9hLoek4rBfzhxvegRG--cBveviNQ

        for file in fileList:
            print(f"file count = {count}")
            print('Title: %s, ID: %s' % (file['title'], file['id']))
            # Get the folder ID that you want
            #fileDownloaded = drive.CreateFile({"id": file["id"]})
            file_title = file['title']
            file.GetContentFile(file["title"])
            dir_path = os.path.dirname(__file__)
            dir_path = dir_path.split("\\")
            for n in range(0,5):
                dir_path.pop()
            dir_path = "\\".join(dir_path)
            filepath = os.path.join(dir_path,file_title)
            with open(filepath,'rb') as f:
                try:
                    client = ExtractorEngineClient()
                    extraction_results = client.extract_crawler(f, filename=file_title)
                    results = extraction_results.json()
                    reg = ExtractedRegulation.objects.filter(document_id=results["results"]["reg_id"]).first()
                    reg.pdf_data.save(
                        file["title"], f
                    )
                    reg.save()
                    file_added += 1
                except:
                    file_not_added += 1
                    pass
            count += 1
            os.remove(filepath)
        print(f"added {file_added} regulations pdf")
        print(f"not add {file_not_added} regulations pdf")

            