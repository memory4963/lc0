import itertools
import multiprocessing as mp
import numpy as np
import random
import sys
sys.path.append('.')
import my_shufflebuffer as sb
import struct
import unittest
import gzip
from select import select

V6_VERSION = struct.pack('i', 6)
V5_VERSION = struct.pack('i', 5)
CLASSICAL_INPUT = struct.pack('i', 1)
V4_VERSION = struct.pack('i', 4)
V3_VERSION = struct.pack('i', 3)
V6_STRUCT_STRING = '4si7432s832sBBBBBBBbfffffffffffffffIHH4H'
V5_STRUCT_STRING = '4si7432s832sBBBBBBBbfffffff'
V4_STRUCT_STRING = '4s7432s832sBBBBBBBbffff'
V3_STRUCT_STRING = '4s7432s832sBBBBBBBb'

def reverse_expand_bits(plane):
    return np.unpackbits(np.array([plane], dtype=np.uint8))[::-1].astype(
        np.float32).tobytes()


def chunk_reader(chunk_filenames, chunk_filename_queue):
    """
    Reads chunk filenames from a list and writes them in shuffled
    order to output_pipes.
    """
    chunks = []
    done = chunk_filenames

    while True:
        if not chunks:
            chunks, done = done, chunks
            random.shuffle(chunks)
        if not chunks:
            print("chunk_reader didn't find any chunks.")
            return None
        while len(chunks):
            filename = chunks.pop()
            done.append(filename)
            chunk_filename_queue.put(filename)
    print("chunk_reader exiting.")
    return None



class ChunkParserInner:
    def __init__(self, chunks, expected_input_format, shuffle_size,
                 sample, diff_focus_min,
                 diff_focus_slope, diff_focus_q_weight, diff_focus_pol_scale):
        """
        Read data and yield batches of raw tensors.

        'parent' the outer chunk parser to store processes. Must not be stored by self directly or indirectly.
        'chunks' list of chunk filenames.
        'shuffle_size' is the size of the shuffle buffer.
        'sample' is the rate to down-sample.
        'diff_focus_min', 'diff_focus_slope', 'diff_focus_q_weight' and 'diff_focus_pol_scale' control diff focus
        'workers' is the number of child workers to use.

        The data is represented in a number of formats through this dataflow
        pipeline. In order, they are:

        chunk: The name of a file containing chunkdata

        chunkdata: type Bytes. Multiple records of v6 format where each record
        consists of (state, policy, result, q)

        raw: A byte string holding raw tensors contenated together. This is
        used to pass data from the workers to the parent. Exists because
        TensorFlow doesn't have a fast way to unpack bit vectors. 7950 bytes
        long.
        """

        self.expected_input_format = expected_input_format

        # Build 2 flat float32 planes with values 0,1
        self.flat_planes = []
        for i in range(2):
            self.flat_planes.append(
                (np.zeros(64, dtype=np.float32) + i).tobytes())

        # set the down-sampling rate
        self.sample = sample
        # set the details for diff focus, defaults accept all positions
        self.diff_focus_min = diff_focus_min
        self.diff_focus_slope = diff_focus_slope
        self.diff_focus_q_weight = diff_focus_q_weight
        self.diff_focus_pol_scale = diff_focus_pol_scale
        # set number of elements in the shuffle buffer.
        self.shuffle_size = shuffle_size
        # Start worker processes, leave 2 for TensorFlow
        self.chunks = chunks

        # if workers is None:
        #     workers = max(1, mp.cpu_count() - 2)

        # if workers > 0:
        #     print("Using {} worker processes.".format(workers))

        #     # Start the child workers running
        #     self.readers = []
        #     self.writers = []
        #     parent.processes = []
        #     self.chunk_filename_queue = mp.Queue(maxsize=4096)
        #     for _ in range(workers):
        #         read, write = mp.Pipe(duplex=False)
        #         p = mp.Process(target=self.task,
        #                        args=(self.chunk_filename_queue, write))
        #         p.daemon = True
        #         parent.processes.append(p)
        #         p.start()
        #         self.readers.append(read)
        #         self.writers.append(write)

        #     parent.chunk_process = mp.Process(target=chunk_reader,
        #                                       args=(chunks,
        #                                             self.chunk_filename_queue))
        #     parent.chunk_process.daemon = True
        #     parent.chunk_process.start()

        self.init_structs()

    def init_structs(self):
        """
        struct.Struct doesn't pickle, so it needs to be separately
        constructed in workers.
        """
        self.v6_struct = struct.Struct(V6_STRUCT_STRING)
        self.v5_struct = struct.Struct(V5_STRUCT_STRING)
        self.v4_struct = struct.Struct(V4_STRUCT_STRING)
        self.v3_struct = struct.Struct(V3_STRUCT_STRING)

    def convert_v6_to_tuple(self, content):
        """
        Unpack a v6 binary record to 5-tuple (state, policy pi, result, q, m)

        v6 struct format is (8356 bytes total):
                                  size         1st byte index
        uint32_t version;                               0
        uint32_t input_format;                          4
        float probabilities[1858];  7432 bytes          8
        uint64_t planes[104];        832 bytes       7440
        uint8_t castling_us_ooo;                     8272
        uint8_t castling_us_oo;                      8273
        uint8_t castling_them_ooo;                   8274
        uint8_t castling_them_oo;                    8275
        uint8_t side_to_move_or_enpassant;           8276
        uint8_t rule50_count;                        8277
        // Bitfield with the following allocation:
        //  bit 7: side to move (input type 3)
        //  bit 6: position marked for deletion by the rescorer (never set by lc0)
        //  bit 5: game adjudicated (v6)
        //  bit 4: max game length exceeded (v6)
        //  bit 3: best_q is for proven best move (v6)
        //  bit 2: transpose transform (input type 3)
        //  bit 1: mirror transform (input type 3)
        //  bit 0: flip transform (input type 3)
        uint8_t invariance_info;                     8278
        uint8_t dep_result;                               8279
        float root_q;                                8280
        float best_q;                                8284
        float root_d;                                8288
        float best_d;                                8292
        float root_m;      // In plies.              8296
        float best_m;      // In plies.              8300
        float plies_left;                            8304
        float result_q;                              8308
        float result_d;                              8312
        float played_q;                              8316
        float played_d;                              8320
        float played_m;                              8324
        // The folowing may be NaN if not found in cache.
        float orig_q;      // For value repair.      8328
        float orig_d;                                8332
        float orig_m;                                8336
        uint32_t visits;                             8340
        // Indices in the probabilities array.
        uint16_t played_idx;                         8344
        uint16_t best_idx;                           8346
        uint64_t reserved;                           8348
        """
        # unpack the V6 content from raw byte array, arbitrarily chose 4 2-byte values
        # for the 8 "reserved" bytes
        (ver, input_format, probs, planes, us_ooo, us_oo, them_ooo, them_oo,
         stm, rule50_count, invariance_info, dep_result, root_q, best_q,
         root_d, best_d, root_m, best_m, plies_left, result_q, result_d,
         played_q, played_d, played_m, orig_q, orig_d, orig_m, visits,
         played_idx, best_idx, reserved1, reserved2, reserved3,
         reserved4) = self.v6_struct.unpack(content)
        """
        v5 struct format was (8308 bytes total)
            int32 version (4 bytes)
            int32 input_format (4 bytes)
            1858 float32 probabilities (7432 bytes)
            104 (13*8) packed bit planes of 8 bytes each (832 bytes)
            uint8 castling us_ooo (1 byte)
            uint8 castling us_oo (1 byte)
            uint8 castling them_ooo (1 byte)
            uint8 castling them_oo (1 byte)
            uint8 side_to_move (1 byte)
            uint8 rule50_count (1 byte)
            uint8 dep_ply_count (1 byte) (unused)
            int8 result (1 byte)
            float32 root_q (4 bytes)
            float32 best_q (4 bytes)
            float32 root_d (4 bytes)
            float32 best_d (4 bytes)
            float32 root_m (4 bytes)
            float32 best_m (4 bytes)
            float32 plies_left (4 bytes)
        """
        # v3/4 data sometimes has a useful value in dep_ply_count (now invariance_info),
        # so copy that over if the new ply_count is not populated.
        if plies_left == 0:
            plies_left = invariance_info
        plies_left = struct.pack('f', plies_left)
        assert input_format == self.expected_input_format

        # Unpack bit planes and cast to 32 bit float
        planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(
            np.float32)
        rule50_divisor = 99.0
        if input_format > 3:
            rule50_divisor = 100.0
        rule50_plane = struct.pack('f', rule50_count / rule50_divisor) * 64

        if input_format == 1:
            middle_planes = self.flat_planes[us_ooo] + \
                            self.flat_planes[us_oo] + \
                            self.flat_planes[them_ooo] + \
                            self.flat_planes[them_oo] + \
                            self.flat_planes[stm]
        elif input_format == 2:
            # Each inner array has to be reversed as these fields are in opposite endian to the planes data.
            them_ooo_bytes = reverse_expand_bits(them_ooo)
            us_ooo_bytes = reverse_expand_bits(us_ooo)
            them_oo_bytes = reverse_expand_bits(them_oo)
            us_oo_bytes = reverse_expand_bits(us_oo)
            middle_planes = us_ooo_bytes + (6*8*4) * b'\x00' + them_ooo_bytes + \
                            us_oo_bytes + (6*8*4) * b'\x00' + them_oo_bytes + \
                            self.flat_planes[0] + \
                            self.flat_planes[0] + \
                            self.flat_planes[stm]
        elif input_format == 3 or input_format == 4 or input_format == 132 or input_format == 5 or input_format == 133:
            # Each inner array has to be reversed as these fields are in opposite endian to the planes data.
            them_ooo_bytes = reverse_expand_bits(them_ooo)
            us_ooo_bytes = reverse_expand_bits(us_ooo)
            them_oo_bytes = reverse_expand_bits(them_oo)
            us_oo_bytes = reverse_expand_bits(us_oo)
            enpassant_bytes = reverse_expand_bits(stm)
            middle_planes = us_ooo_bytes + (6*8*4) * b'\x00' + them_ooo_bytes + \
                            us_oo_bytes + (6*8*4) * b'\x00' + them_oo_bytes + \
                            self.flat_planes[0] + \
                            self.flat_planes[0] + \
                            (7*8*4) * b'\x00' + enpassant_bytes

        # Concatenate all byteplanes. Make the last plane all 1's so the NN can
        # detect edges of the board more easily
        aux_plus_6_plane = self.flat_planes[0]
        if (input_format == 132
                or input_format == 133) and invariance_info >= 128:
            aux_plus_6_plane = self.flat_planes[1]
        planes = planes.tobytes() + \
                 middle_planes + \
                 rule50_plane + \
                 aux_plus_6_plane + \
                 self.flat_planes[1]

        assert len(planes) == ((8 * 13 * 1 + 8 * 1 * 1) * 8 * 8 * 4)

        if ver == V6_VERSION:
            winner = struct.pack('fff', 0.5 * (1.0 - result_d + result_q),
                                 result_d, 0.5 * (1.0 - result_d - result_q))
        else:
            dep_result = float(dep_result)
            assert dep_result == 1.0 or dep_result == -1.0 or dep_result == 0.0
            winner = struct.pack('fff', dep_result == 1.0, dep_result == 0.0,
                                 dep_result == -1.0)

        best_q_w = 0.5 * (1.0 - best_d + best_q)
        best_q_l = 0.5 * (1.0 - best_d - best_q)
        assert -1.0 <= best_q <= 1.0 and 0.0 <= best_d <= 1.0
        best_q = struct.pack('fff', best_q_w, best_d, best_q_l)

        return (planes, probs, winner, best_q, plies_left)

    def sample_record(self, chunkdata):
        """
        Randomly sample through the v3/4/5/6 chunk data and select records in v6 format
        Downsampling to avoid highly correlated positions skips most records, and 
        diff focus may also skip some records.
        """
        version = chunkdata[0:4]
        if version == V6_VERSION:
            record_size = self.v6_struct.size
        elif version == V5_VERSION:
            record_size = self.v5_struct.size
        elif version == V4_VERSION:
            record_size = self.v4_struct.size
        elif version == V3_VERSION:
            record_size = self.v3_struct.size
        else:
            return

        records = []
        for i in range(0, len(chunkdata), record_size):
            # if self.sample > 1:
            #     # Downsample, using only 1/Nth of the items.
            #     if random.randint(0, self.sample - 1) != 0:
            #         continue  # Skip this record.

            record = chunkdata[i:i + record_size]
            # for earlier versions, append fake bytes to record to maintain size
            if version == V3_VERSION:
                # add 16 bytes of fake root_q, best_q, root_d, best_d to match V4 format
                record += 16 * b'\x00'
            if version == V3_VERSION or version == V4_VERSION:
                # add 12 bytes of fake root_m, best_m, plies_left to match V5 format
                record += 12 * b'\x00'
                # insert 4 bytes of classical input format tag to match v5 format
                record = record[:4] + CLASSICAL_INPUT + record[4:]
            if version == V3_VERSION or version == V4_VERSION or version == V5_VERSION:
                # add 48 byes of fake result_q, result_d etc
                record += 48 * b'\x00'

            if version == V6_VERSION:
                # diff focus code, peek at best_q, orig_q and pol_kld from record (unpacks as tuple with one item)
                best_q = struct.unpack('f', record[8284:8288])[0]
                orig_q = struct.unpack('f', record[8328:8332])[0]
                pol_kld = struct.unpack('f', record[8348:8352])[0]

                # if orig_q is NaN or pol_kld is 0, accept, else accept based on diff focus
                if not np.isnan(orig_q) and pol_kld > 0:
                    diff_q = abs(best_q - orig_q)
                    q_weight = self.diff_focus_q_weight
                    pol_scale = self.diff_focus_pol_scale
                    total = (q_weight * diff_q + pol_kld) / (q_weight +
                                                             pol_scale)
                    thresh_p = self.diff_focus_min + self.diff_focus_slope * total
                    if thresh_p < 1.0 and random.random() > thresh_p:
                        continue

            records.append(record)
        return records

    def single_file_gen(self, filename):
        try:
            with gzip.open(filename, 'rb') as chunk_file:
                version = chunk_file.read(4)
                chunk_file.seek(0)
                if version == V6_VERSION:
                    record_size = self.v6_struct.size
                elif version == V5_VERSION:
                    record_size = self.v5_struct.size
                elif version == V4_VERSION:
                    record_size = self.v4_struct.size
                elif version == V3_VERSION:
                    record_size = self.v3_struct.size
                else:
                    print('Unknown version {} in file {}'.format(
                        version, filename))
                    return
                while True:
                    chunkdata = chunk_file.read(256 * record_size)
                    if len(chunkdata) == 0:
                        break
                    return self.sample_record(chunkdata)
        except:
            print("failed to parse {}".format(filename))

    def sequential_gen(self):
        for filename in self.chunks:
            for item in self.single_file_gen(filename):
                yield item

    def sequential(self):
        gen = self.sequential_gen()  # read from all files in order in this process.
        gen = self.tuple_gen(gen)  # convert v6->tuple
        gen = self.batch_gen(gen, allow_partial=False)  # assemble into batches
        for b in gen:
            yield b

    def task(self, filename):
        """
        Run in fork'ed process, read data from chunkdatasrc, parsing, shuffling and
        sending v6 data through pipe back to main process.
        """
        data = []
        for item in self.single_file_gen(filename):
            data.append(item)
        return data

    def v6_gen(self):
        """
        Read v6 records from child workers, shuffle, and yield
        records.
        """
        sbuff = sb.ShuffleBuffer(self.v6_struct.size, self.shuffle_size)

        reader_tags = {r: i for i, r in enumerate(self.readers)}
        order_counters = {r: itertools.count() for r in self.readers}

        while len(self.readers):
            for r in self.readers:
                try:
                    s = r.recv_bytes()
                    s = (s, reader_tags[r], next(order_counters[r]))
                    s = sbuff.insert_or_replace(s)
                    if s is None:
                        continue  # shuffle buffer not yet full
                    yield s
                except EOFError:
                    print("Reader EOF")
                    self.readers.remove(r)
        # drain the shuffle buffer.
        while True:
            s = sbuff.extract()
            if s is None:
                return
            yield s

    def tuple_gen(self, gen):
        """
        Take a generator producing v6 records and convert them to tuples.
        applying a random symmetry on the way.
        """
        for r in gen:
            yield self.convert_v6_to_tuple(r)

    def batch_gen(self, gen, allow_partial=True):
        """
        Pack multiple records into a single batch
        """
        # Get N records. We flatten the returned generator to
        # a list because we need to reuse it.
        while True:
            s = list(itertools.islice(gen, self.batch_size))
            if not len(s) or (not allow_partial and len(s) != self.batch_size):
                return
            yield (b''.join([x[0] for x in s]), b''.join([x[1] for x in s]),
                   b''.join([x[2] for x in s]), b''.join([x[3] for x in s]),
                   b''.join([x[4] for x in s]))

    def parse(self):
        """
        Read data from child workers and yield batches of unpacked records
        """
        gen = self.v6_gen()  # read from workers
        gen = self.tuple_gen(gen)  # convert v6->tuple
        gen = self.batch_gen(gen)  # assemble into batches
        for b in gen:
            yield b