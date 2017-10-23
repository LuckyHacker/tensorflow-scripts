import sys

csv_file_path = "data/access.csv"
log_file_path = "data/access.log"
allowed_labels = (0, 1)

def read_single_keypress():
    # https://stackoverflow.com/a/6599441
    """Waits for a single keypress on stdin.

    This is a silly function to call if you need to do it a lot because it has
    to store stdin's current setup, setup stdin for reading single keystrokes
    then read the single keystroke then revert stdin back after reading the
    keystroke.

    Returns the character of the key that was pressed (zero on
    KeyboardInterrupt which can happen when a signal gets handled)

    """
    import termios, fcntl, sys, os
    fd = sys.stdin.fileno()
    # save old state
    flags_save = fcntl.fcntl(fd, fcntl.F_GETFL)
    attrs_save = termios.tcgetattr(fd)
    # make raw - the way to do this comes from the termios(3) man page.
    attrs = list(attrs_save) # copy the stored version to update
    # iflag
    attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK
                  | termios.ISTRIP | termios.INLCR | termios. IGNCR
                  | termios.ICRNL | termios.IXON )
    # oflag
    attrs[1] &= ~termios.OPOST
    # cflag
    attrs[2] &= ~(termios.CSIZE | termios. PARENB)
    attrs[2] |= termios.CS8
    # lflag
    attrs[3] &= ~(termios.ECHONL | termios.ECHO | termios.ICANON
                  | termios.ISIG | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # turn off non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags_save & ~os.O_NONBLOCK)
    # read a single keystroke
    try:
        ret = sys.stdin.read(1) # returns a single character
    except KeyboardInterrupt:
        ret = 0
    finally:
        # restore old state
        termios.tcsetattr(fd, termios.TCSAFLUSH, attrs_save)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save)
    return ret


def read_files():
    with open(csv_file_path, "r") as f:
        csv_lines = f.readlines()

    with open(log_file_path, "r") as f:
        log_lines = f.readlines()

    return csv_lines, log_lines

def label_all(csv_lines, log_lines):
    with open(csv_file_path, "w") as f:
        f.write(csv_lines[0].replace("\n", "") + ",label\n")
        for i in range(len(log_lines)):
            sys.stdout.write(log_lines[i])
            label = read_single_keypress()
            sys.stdout.write("\n")

            if label not in list(map(lambda x: str(x), allowed_labels)):
                break

            f.write(csv_lines[i + 1].replace("\n", "") + ",{}\n".format(label))

if __name__ == "__main__":
    csv_lines, log_lines = read_files()
    label_all(csv_lines, log_lines)
