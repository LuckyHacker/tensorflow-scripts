import paramiko
import apache_log_parser


host = "192.168.1.5"

def fetch_access_log():
    paramiko.util.log_to_file('/tmp/paramiko.log')

    port = 22
    transport = paramiko.Transport((host, port))

    username = "root"
    password = "root66"
    transport.connect(username=username, password=password)

    sftp = paramiko.SFTPClient.from_transport(transport)

    filepath = '/var/log/apache2/access.log'
    localpath = 'data/access.log'
    sftp.get(filepath, localpath)

    sftp.close()
    transport.close()


def parse_log():
    parsed_lines = []
    line_parser = apache_log_parser.make_parser('%h %l %u %t \"%r\" %>s %b \"%{Referer}i\" \"%{User-Agent}i')
    with open("data/access.csv", "w") as csv:
        csv.write("date,time,remote_host,request_method,request_url,status\n")
        with open("data/access.log", "r") as f:
            for line in f.readlines():
                parsed_line = line_parser(line)
                date, time = parsed_line["time_received_isoformat"].split("T")
                csv_line = "{},{},{},{},{},{}\n".format(
                        date,
                        time,
                        parsed_line["remote_host"],
                        parsed_line["request_method"],
                        parsed_line["request_url"],
                        parsed_line["status"])

                csv.write(csv_line)



if __name__ == "__main__":
    fetch_access_log()
    parse_log()
