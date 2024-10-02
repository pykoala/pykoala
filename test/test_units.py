#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import os
import pykoala.data_container as dc 
import numpy as np

@pytest.fixture(scope="session")
def session_tmpdir(tmpdir_factory):
    session_dir = tmpdir_factory.mktemp("session_tmp")
    return session_dir
    
class TestDataContainer:

    def test_history_record(self):
        t_hist_record = dc.HistoryRecord("Title for test",["comments","for","test"])
        t_hist_record_tostr = dc.HistoryRecord(["Title", "for", "test"],["comments","for","test"])
        
        assert t_hist_record.title == "Title for test"
        assert not t_hist_record.title == "This is not the title, that is why we use 'not'"
        assert t_hist_record.comments == ["comments","for","test"]
        assert not t_hist_record.comments == ["wrong","tag"]       
        assert type(t_hist_record_tostr.title) == list
        assert type(t_hist_record_tostr.to_str()) == str


    def test_data_container_history(self,session_tmpdir):
        list_of_entries = [['TITLE','COMMENTS','TAG'],['title_1','title_2','title_3']]

        log = dc.DataContainerHistory()
        log_expanded = dc.DataContainerHistory()
        log.initialise_record(list_of_entries=list_of_entries)
        log_expanded.initialise_record(list_of_entries=list_of_entries)
        
        log_expanded.log_record("NEW_TITLE", ['new_title_1'])
        _tmp_file = session_tmpdir.join("session_file.txt")
        _tmp_header = log.dump_to_header()
        log.dump_to_text(_tmp_file)
        _title_from_entry = log.get_entries_from_header(_tmp_header)[0].title
        _comments_from_entry = log.get_entries_from_header(_tmp_header)[0].comments
        


        #Commented line also works, but it is too confusing to read:
        #assert file.readlines()[0].rstrip('\n') == list_of_entries[0][0]+": "+list_of_entries[0][1] 
        assert _tmp_file.readlines()[0].rstrip('\n') == "TITLE: COMMENTS", "dump_to_text() does not write correct record"
        assert log.is_record("TITLE"), "is_record() cannot find correct record"
        assert not log.is_record("INCORRECT TITLE"), "is_record() takes incorrect record as correct"
        assert _tmp_header[0] == "COMMENTS", "dump_to_header() does not save correct data in astropy.fits.Header"
        assert log_expanded.is_record("NEW_TITLE"), "log_record() does not append correct data in record" 
        assert _title_from_entry == 'TITLE', "get_entries_from_header() gets wrong title"
        assert _comments_from_entry[0] == 'COMMENTS', "get_entries_from_header() gets wrong comments"


    def test_data_mask(self):
