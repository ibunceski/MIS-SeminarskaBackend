from datetime import date, timedelta


class DataDateChecker:
    def __init__(self, storage):
        self.storage = storage

    def get_last_data_date(self, issuer):
        last_date = self.storage.get_issuer_date(issuer)

        if last_date:
            return last_date
        else:
            return date.today() - timedelta(days=3650)
