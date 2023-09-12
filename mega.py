from typing import Tuple
import os
import subprocess

PATH_TO_MEGA_CMD_DIR = r"" #E.G C:\Users\someuser\AppData\Local\MEGAcmd





def login(email,password) -> Tuple[bool,int]:
    target = os.path.join(PATH_TO_MEGA_CMD_DIR,"mega-login.bat")
    result = subprocess.call(f"{target} {email} {password}")
    if result == 0:
        #logged in
        return (True,0)
    if result == 54:
        #already logged in 
        return (False,54)
    else:
        return (False,result)

def logout() -> None:
    target = os.path.join(PATH_TO_MEGA_CMD_DIR,"mega-logout.bat")
    result = subprocess.call(f"{target}")
    
def who_am_i() -> str | None: #get logged in user email (None if logged out)
    try:
        target = os.path.join(PATH_TO_MEGA_CMD_DIR,"mega-whoami.bat")
        output = subprocess.check_output([target],text=True)

        #parse output to only email
        return output.removeprefix("Account e-mail: ").strip()

    except subprocess.CalledProcessError as e:
        if e.returncode == 57:
            #not logged in.
            return None
        else:
            raise e
        

def put(fp:str): #upload a file
    try:
        target = os.path.join(PATH_TO_MEGA_CMD_DIR,"mega-put.bat")
        output = subprocess.call([target,fp])
    
    except subprocess.CalledProcessError as e:
        raise e




#look into webadv server (this seems to be most suitable for syncinc recording live to db)



if __name__ == "__main__":
    pass


