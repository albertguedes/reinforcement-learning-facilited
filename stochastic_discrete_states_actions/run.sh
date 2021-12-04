# import the modules
import os, csv
 
def write(first, last, phone):
 
    # checks if the csv file already exists or not
    if os.path.isfile('data.csv'):
        with open('data.csv', 'a') as file:
 
            # if it exists then appending the rows to it
            writer = csv.writer(file)
 
            # writing the rows
            writer.writerow([first, last, phone])
    else:
 
        # column names
        header = ['First Name', 'Last Name', 'Phone']
 
        # creating new file in write (w) mode
        with open('data.csv', 'w') as file:
 
            # initializing writer
            writer = csv.DictWriter(file, fieldnames = header)
 
            # writing the column header
            writer.writeheader()
 
            # writing the new entry
            writer.writerow({
                'First Name' : first,
                'Last Name' : last,
                'Phone' : phone})
 
# driver function
if __name__ == "__main__":
 
    # reads the data from txt file
    # removes new line and split it with '|' delimiter
    data = open('data.txt').readline().strip().split('|')
 
    # unziping the list to individual variables
    first_name, last_name, phone = data
 
    # calling write function
    write(first_name, last_name, phone)
 
    # removes the data.txt (optional)
    os.remove('data.txt')
