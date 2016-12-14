import logging

from base import BaseRepresentation as Base

from os import listdir
from os.path import isfile, isdir, join


def get_dump_size(path):
    with open(path) as dump:
        dump.seek(0, 2)
        return dump.tell()


class MemorySliceGenerator:
    file_paths = []
    logger = logging.getLogger("MemorySliceGenerator")

    def __init__(self, paths, size, base=Base.bit):
        """
        Creates a new Memory Slice Generator.
        :param paths: list of paths to files or directories to read from.
        :param size: size of outputted X and y.
        :param base: BaseRepresentation of the outputted X and y.
        """
        self.get_file_list(paths)
        self.size = size
        self.base = base
        self.logger.info("MemorySliceGenerator size: %d, base: %s, initialized with %d files", size, base.name,
                         len(self.file_paths))

    def get_file_list(self, paths):
        for path in paths:
            if isfile(path):
                self.file_paths.append(path)
            elif isdir(path):
                self.get_file_list([join(path, p) for p in listdir(path)])

    def read_slice(self, dump):
        read_size = int(self.size * self.base.value / Base.byte.value)
        chunk = list(dump.read(read_size))
        if len(chunk) < read_size:
            self.logger.debug("Read %d bytes from %s, expected %d", len(chunk), dump.name, read_size)
            return 0
        return self.string_to_base_array(chunk)

    def string_to_base_array(self, string):
        if self.base == Base.bit:
            return [int(digit) for character in string for digit in format(ord(character), '08b')]
        if self.base == Base.byte:
            return [ord(character) for character in string]

    def generate_memory_slices(self):
        while 1:
            for file_path in self.file_paths:
                self.logger.info("Opened dump " + file_path)
                with open(file_path, 'rb') as dump:
                    chunk_counter = 0
                    chunk = self.read_slice(dump)
                    chunk_size = len(chunk)
                    while chunk:
                        chunk_counter += 1
                        self.logger.debug("Received %d'th chunk of size %d %ss", chunk_counter, len(chunk),
                                          self.base.name)
                        yield chunk, chunk
                        chunk = self.read_slice(dump)
                    self.logger.info("Finished reading dump - %s", dump.name)
                    self.logger.info("%d chunks of size %d %ss were read", chunk_counter, chunk_size, self.base.name)
                    self.logger.info("overall %d %ss were read out of %d %ss from dump %s", chunk_counter * chunk_size,
                                     self.base.name, get_dump_size(file_path) * Base.byte.value / self.base.value,
                                     self.base.name, dump.name)
                    self.logger.info("Moving to next dump")
