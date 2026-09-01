---
id: herdr
aliases: []
tags: []
---
# Herdr Terminal Multiplexer Overview and Commands 


## Basic Commands: 

Prefix: Ctrl-b (like tmux)

You can exit a session using prefix q / connect again with herdr 

You can stop the server using: herdr server stop -> it shuts down all the processes and agents. 
However, herdr considers session states (workspaces, tabs etc,) different from process states. 

**Config File**

The config file lives here: ~/.config/herdr/config.toml and you can populate it with the default config using herdr --default-config 



**Settings**

You can easily define themes, sounds and notifications. 


**Sessions**

You can create a new session with: herdr session name 

List current sessions: herdr session list 
Stop session: herdr session stop && hrdr session delete name 

**Remote session**

You can use herdr remote mode to connect to a remote env but keeping your current setup for herdr using: 
herdr --remote server_name (It requires herdr to be installed there to)

Connect to a remote session with --session session_name 


**Custom key commands**



**Installing a plugin**

herdr plugin install name_plugin 

herdr plugin list 

