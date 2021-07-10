import csv
import os

input_csv = 'aaa.csv'
output_dir = 'result'
output_prefix = 'splitted_aaa'

os.makedirs(output_dir, exist_ok=True)

max_batch_size = 2
batch_id = 0

def get_csv_out(batch_id, output_prefix, output_dir):
    # Get file write object for the output csv
    output_csv = '{output_prefix}_{batch_id}.csv'.format(output_prefix=output_prefix, batch_id=batch_id)
    csv_out = open(os.path.join(output_dir, output_csv), 'w')
    return csv_out
    

with open(input_csv, 'r') as csv_in:
    csv_reader = csv.reader(csv_in, delimiter=',')
    header = next(csv_reader)
    print('header: {}'.format(header))
    cur_batch_size = 0
    csv_out = get_csv_out(batch_id, output_prefix, output_dir)
    csv_writer = csv.writer(csv_out, delimiter=',')
    print("Start writing file number: {}".format(batch_id))
    csv_writer.writerow(header)
    for row in csv_reader:
        if cur_batch_size >= max_batch_size:
            # The batch is full, create new batch (new file)
            csv_out.close()
            batch_id += 1
            csv_out = get_csv_out(batch_id, output_prefix, output_dir)
            csv_writer = csv.writer(csv_out, delimiter=',')
            print("Start writing file number: {}".format(batch_id))
            csv_writer.writerow(header)
            cur_batch_size = 0
        csv_writer.writerow(row)
        cur_batch_size += 1
    csv_out.close()




    
