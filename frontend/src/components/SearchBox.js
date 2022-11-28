import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function SearchBox() {
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const submitHandler = (e) => {
    e.preventDefault();
    navigate(`/search/name/${name}`);
  };
  return (
    <form className="search" onSubmit={submitHandler} data-testid= "searchbox-test">
      <div className="row">
        <input
          type="text"
          name="q"
          id="q"
          data-testid = "search-test"
          onChange={(e) => setName(e.target.value)}
        ></input>
        <button className="primary" type="submit" data-testid = "bttn-test">
          <i className="fa fa-search"></i>
        </button>
      </div>
    </form>
  );
}
