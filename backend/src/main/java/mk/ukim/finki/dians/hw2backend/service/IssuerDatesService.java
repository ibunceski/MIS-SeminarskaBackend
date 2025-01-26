package mk.ukim.finki.dians.hw2backend.service;

import mk.ukim.finki.dians.hw2backend.model.IssuerDates;

import java.util.Optional;

public interface IssuerDatesService {

    public Optional<IssuerDates> getLastDateForIssuer(String issuer);
}