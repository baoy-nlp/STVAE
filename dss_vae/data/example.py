class Example(object):
    def __init__(self, idx=0, key_field='src', fields=None, values=None, **kwargs):
        if fields is not None:
            for field, value in zip(fields, values):
                self.__setattr__(field, value)
            self.fields = fields
        else:
            self.fields = []
            for key, value in kwargs.items():
                self.__setattr__(key, value)
                self.fields.append(key)
        self.key_field = key_field
        self.idx = idx

    @staticmethod
    def load(raw, idx=0, fields=None, spliter='\t', seg=' '):
        vals = raw.strip().split(spliter)
        if len(vals) <= 1:  # vae
            return Example(idx=idx, src=vals[0].split(seg), tgt=None, )
        elif len(vals) == 2:  # dss-vae
            return Example(idx=idx, src=vals[0].split(seg), tgt=vals[1].split(seg))
        else:
            assert fields is not None, 'need indicate the fields name'
            return Example(idx=idx, fields=fields, values=[item.split(seg) for item in vals])

    def __str__(self):
        return " ".join(getattr(self, self.key_field, ""))

    def __len__(self):
        return len(getattr(self, self.key_field, ""))


task_to_example = {
    "DSS-VAE": Example,
    "Para-VAE": Example
}
