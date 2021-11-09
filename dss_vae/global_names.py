class FileNames(object):
    tree_suffix = ".clean"

    @staticmethod
    def update_names(args):
        FileNames.tree_suffix = getattr(args, 'tree_suffix', '.clean')
