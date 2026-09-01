import io
p="backend/server.py"
s=io.open(p,encoding="utf-8").read()
old = """    except Exception:
        return ""




def _client_ip"""
new =  except Exception:
